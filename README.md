# Text Slider: Efficient and Plug-and-Play Continuous Concept Control for Image/Video Synthesis via LoRA Adapters

**WACV 2026**

[Pin-Yen Chiu](https://itsnickchiu.github.io), 
[I-Sheng Fang](https://ishengfang.github.io), 
[Jun-Cheng Chen](https://homepage.citi.sinica.edu.tw/pages/pullpull/index_en.html)

Research Center for Information Technology Innovation, Academia Sinica

#### [Project Page](https://textslider.github.io) | [Paper](https://arxiv.org/pdf/2509.18831) | [arXiv](https://arxiv.org/abs/2509.18831)

<img src="teaser.png" width="80%"></img>

## Environment setup
```
conda create -n textslider python=3.10 -y
conda activate textslider
pip install -r requirements.txt
```

## Inference
We provide ready-to-use notebooks so you can try Text Slider immediately:
* Stable Diffusion XL: [SDXL-inference.ipynb](https://github.com/aiiu-lab/TextSlider/blob/master/SDXL-inference.ipynb)
* Stable Diffusion 1: [SD1-inference.ipynb](https://github.com/aiiu-lab/TextSlider/blob/master/SD1-inference.ipynb)

We also include several pre-trained slider checkpoints in the `models/` directory.
Feel free to try with your own prompts and attributes!

## Training
### Generate Training Prompts
We follow the prompt-generation method from [Concept Slider](https://github.com/rohitgandikota/sliders) (see their [GPT_prompt_helper.ipynb](https://github.com/rohitgandikota/sliders/blob/main/GPT_prompt_helper.ipynb)), which requires an OpenAI API key.
If you prefer not to use the API, we provide a standalone system prompt in `trainscript/prompt_generate.txt`.
Simply copy–paste it into ChatGPT, provide the attribute you want to train, and it will generate:
* target
* positive
* negative
* preservation

You can then use these to create a new `prompt.yaml` following our existing format.

### Train a Slider
Below is an example for training an `age` slider.

First, move into the training directory:
```
cd trainscript
```
Then paste the text after `"preservation"` (generated from ChatGPT) into the `--attributes` argument:
```
python train_text_lora.py \
  --attributes "white race, black race, indian race, asian race, hispanic race ; male, female" \
  --name "ageslider" \
  --rank 4 \
  --alpha 1 \
  --config_file "data/config.yaml" \
  --prompts_file "data/prompts-age.yaml"
```

### Citation
If you find our work useful, please consider cite this work as
```bibtex
@inproceedings{chiu2026textslider,
  title={Text Slider: Efficient and Plug-and-Play Continuous Concept Control for Image/Video Synthesis via LoRA Adapters},
  author={Pin-Yen Chiu and I-Sheng Fang and Jun-Cheng Chen},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2026}
}
```