import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import copy
import os

import utils.prompt_util as prompt_util
from utils.prompt_util import PromptEmbedsPair, PromptSettings
import utils.config_util as config_util
from utils.config_util import RootConfig

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers import StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device: int
):
    save_path = Path(config.save.path)
    weight_dtype = config_util.parse_precision(config.train.precision)

    # CLIP
    model_name = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")

    # the original one
    ori_text_enc_1 = copy.deepcopy(text_encoder_1)
    ori_text_enc_1.to(device, dtype=weight_dtype)
    ori_text_enc_1.requires_grad_(False)
    ori_text_enc_1.eval()

    # the lora one
    text_encoder_1.to(device, dtype=weight_dtype)
    text_encoder_1.requires_grad_(False)
    text_encoder_1.eval()

    # OpenCLIP
    model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_name, subfolder="text_encoder_2")

    # the original one
    ori_text_enc_2 = copy.deepcopy(text_encoder_2)
    ori_text_enc_2.to(device, dtype=weight_dtype)
    ori_text_enc_2.requires_grad_(False)
    ori_text_enc_2.eval()

    # the lora one
    text_encoder_2.to(device, dtype=weight_dtype)
    text_encoder_2.requires_grad_(False)
    text_encoder_2.eval()

    # inject lora into text encoders
    text_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        init_lora_weights=True,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    text_encoder_1.add_adapter(text_lora_config)
    text_encoder_2.add_adapter(text_lora_config)
    lora_layers = filter(lambda p: p.requires_grad, list(text_encoder_1.parameters())+list(text_encoder_2.parameters()))

    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]
    ori_text_encoders = [ori_text_enc_1, ori_text_enc_2]

    optimizer = torch.optim.AdamW(lora_layers, lr=config.train.lr)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1)
    criteria = torch.nn.MSELoss()

    cache_text_enc_tokenwise = {}
    cache_text_enc_pooled = {}
    cache_lora_tokenwise = {}
    cache_lora_pooled = {}
    prompt_pairs_tokenwise: list[PromptEmbedsPair] = []
    prompt_pairs_pooled: list[PromptEmbedsPair] = []

    with torch.no_grad():
        # precompute text embeddings from "original text encoders"
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                h_s = []
                p_s = []

                # CLIP
                text_tokens = tokenizer_1(
                            prompt,
                            padding="max_length",
                            max_length=tokenizer_1.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids
                out = ori_text_enc_1(text_tokens.to(device))
                h_s.append(out.last_hidden_state)
                p_s.append(out.pooler_output)

                # OpenCLIP
                text_tokens = tokenizer_2(
                            prompt,
                            padding="max_length",
                            max_length=tokenizer_2.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids
                out = ori_text_enc_2(text_tokens.to(device))
                h_s.append(out.last_hidden_state)
                p_s.append(out.text_embeds)

                # concate the tokenwise and pooled embeddings repectively
                cache_text_enc_tokenwise[prompt] = torch.concat(h_s, dim=-1)
                cache_text_enc_pooled[prompt] = torch.concat(p_s, dim=-1)

            prompt_pairs_tokenwise.append(
                PromptEmbedsPair(
                    criteria,
                    cache_text_enc_tokenwise[settings.target],
                    cache_text_enc_tokenwise[settings.positive],
                    cache_text_enc_tokenwise[settings.unconditional],
                    cache_text_enc_tokenwise[settings.neutral],
                    settings,
                )
            )
            prompt_pairs_pooled.append(
                PromptEmbedsPair(
                    criteria,
                    cache_text_enc_pooled[settings.target],
                    cache_text_enc_pooled[settings.positive],
                    cache_text_enc_pooled[settings.unconditional],
                    cache_text_enc_pooled[settings.neutral],
                    settings,
                )
            )

    pbar = tqdm(range(config.train.iterations))
    for i in pbar:
        lora_prompt_pairs_tokenwise: list[PromptEmbedsPair] = []
        lora_prompt_pairs_pooled: list[PromptEmbedsPair] = []

        # extract text embeddings from the text encoders + LoRA
        for settings in prompts:
            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                h_s = []
                p_s = []

                # CLIP
                text_tokens = tokenizer_1(
                            prompt,
                            padding="max_length",
                            max_length=tokenizer_1.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids
                out = text_encoder_1(text_tokens.to(device))
                h_s.append(out.last_hidden_state)
                p_s.append(out.pooler_output)

                # OpenCLIP
                text_tokens = tokenizer_2(
                            prompt,
                            padding="max_length",
                            max_length=tokenizer_2.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        ).input_ids
                out = text_encoder_2(text_tokens.to(device))
                h_s.append(out.last_hidden_state)
                p_s.append(out.text_embeds)

                # concate the tokenwise and pooled embeddings repectively
                cache_lora_tokenwise[prompt] = torch.concat(h_s, dim=-1)
                cache_lora_pooled[prompt] = torch.concat(p_s, dim=-1)
            
            lora_prompt_pairs_tokenwise.append(
                PromptEmbedsPair(
                    criteria,
                    cache_lora_tokenwise[settings.target],
                    cache_lora_tokenwise[settings.positive],
                    cache_lora_tokenwise[settings.unconditional],
                    cache_lora_tokenwise[settings.neutral],
                    settings,
                )
            )
            lora_prompt_pairs_pooled.append(
                PromptEmbedsPair(
                    criteria,
                    cache_lora_pooled[settings.target],
                    cache_lora_pooled[settings.positive],
                    cache_lora_pooled[settings.unconditional],
                    cache_lora_pooled[settings.neutral],
                    settings,
                )
            )

        optimizer.zero_grad()

        idx = torch.randint(0, len(prompt_pairs_tokenwise), (1,)).item()
        prompt_pair_tokenwise: PromptEmbedsPair = prompt_pairs_tokenwise[idx]
        prompt_pair_pooled: PromptEmbedsPair = prompt_pairs_pooled[idx]
        lora_prompt_pair_tokenwise: PromptEmbedsPair = lora_prompt_pairs_tokenwise[idx]
        lora_prompt_pair_pooled: PromptEmbedsPair = lora_prompt_pairs_pooled[idx]

        loss_tokenwise = prompt_pair_tokenwise.loss(
            target_latents=lora_prompt_pair_tokenwise.target,
            positive_latents=prompt_pair_tokenwise.positive,
            neutral_latents=prompt_pair_tokenwise.neutral,
            unconditional_latents=prompt_pair_tokenwise.unconditional,
        )
        loss_pooled = prompt_pair_pooled.loss(
            target_latents=lora_prompt_pair_pooled.target,
            positive_latents=prompt_pair_pooled.positive,
            neutral_latents=prompt_pair_pooled.neutral,
            unconditional_latents=prompt_pair_pooled.unconditional,
        )
        loss = loss_tokenwise + loss_pooled

        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    print("Saving Checkpoint...")
    save_path.mkdir(parents=True, exist_ok=True)

    for text_encoder, lora_name in zip(text_encoders, ['clip', 'openclip']):
        ckpt_save_path = os.path.join(save_path, f"{lora_name}_lora")
        text_enc_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
        StableDiffusionPipeline.save_lora_weights(
            save_directory=ckpt_save_path,
            text_encoder_lora_layers=text_enc_lora_state_dict,
            safe_serialization=True,
        )

    print("Done.")


def main(args):
    config_file = args.config_file
    config = config_util.load_config_from_yaml(config_file)
    torch.manual_seed(config.train.seed)

    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]

    if args.prompts_file is not None:
        config.prompts_file = args.prompts_file

    config.save.name += f'_rank{args.rank}'
    config.save.path += f'/{config.save.name}'
    
    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    # print(prompts)
    device = torch.device(f"cuda:{args.device}")

    print(f"=== Configs ===")
    print(f"seed: {config.train.seed}")
    print(f"lr: {config.train.lr}")
    print(f"rank: {args.rank}")
    print(f"save_path: {config.save.path}\n")
    
    train(config, prompts, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
    )
    parser.add_argument(
        "--prompts_file",
        required=False,
        help="Prompts file for training.",
        default=None
    )
    # config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=None,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=None,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default=None,
        help="Device to train on.",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default=None,
        help="attritbutes to disentangle (comma seperated string)",
    )
    # --attributes 'male, female'
    
    args = parser.parse_args()

    main(args)
