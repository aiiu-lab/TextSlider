from typing import Literal, Optional, Union, List
import yaml
from pydantic import BaseModel
import torch
import copy

ACTION_TYPES = Literal[
    "erase",
    "enhance",
]

class PromptSettings(BaseModel):
    target: str
    positive: str = None
    unconditional: str = ""
    neutral: str = None
    action: ACTION_TYPES = "erase"


class PromptEmbedsPair:
    target: torch.FloatTensor
    positive: torch.FloatTensor
    unconditional: torch.FloatTensor
    neutral: torch.FloatTensor

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: torch.FloatTensor,
        positive: torch.FloatTensor,
        unconditional: torch.FloatTensor,
        neutral: torch.FloatTensor,
        settings: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral
        self.action = settings.action

    def _erase(
        self,
        target_latents: torch.FloatTensor,
        positive_latents: torch.FloatTensor,
        unconditional_latents: torch.FloatTensor,
        neutral_latents: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Target latents are going not to have the positive concept."""
        return self.loss_fn(
            target_latents,
            neutral_latents
            - (positive_latents - unconditional_latents)
        )
    
    def _enhance(
        self,
        target_latents: torch.FloatTensor,
        positive_latents: torch.FloatTensor,
        unconditional_latents: torch.FloatTensor,
        neutral_latents: torch.FloatTensor,
    ):
        """Target latents are going to have the positive concept."""

        return self.loss_fn(
            target_latents,
            neutral_latents
            + (positive_latents - unconditional_latents)
        )

    def loss(
        self,
        **kwargs,
    ):
        if self.action == "erase":
            return self._erase(**kwargs)

        elif self.action == "enhance":
            return self._enhance(**kwargs)

        else:
            raise ValueError("action must be erase or enhance")


def load_prompts_from_yaml(path, attributes = []):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    print("=== Original Input Prompts ===")
    print(prompts)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")
    if len(attributes)!=0:
        newprompts = []
        for i in range(len(prompts)):
            for att in attributes:
                copy_ = copy.deepcopy(prompts[i])
                copy_['target'] = att + ' ' + copy_['target']
                copy_['positive'] = att + ' ' + copy_['positive']
                copy_['neutral'] = att + ' ' + copy_['neutral']
                copy_['unconditional'] = att + ' ' + copy_['unconditional']
                newprompts.append(copy_)
    else:
        newprompts = copy.deepcopy(prompts)
    
    print("\n=== New Prompts ===")
    print(newprompts)
    print(f"\nnumber of prompts: {len(prompts)}")
    print(f"number of new prompts: {len(newprompts)}")
    prompt_settings = [PromptSettings(**prompt) for prompt in newprompts]

    return prompt_settings
