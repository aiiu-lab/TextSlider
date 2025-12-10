from typing import Literal, Optional
import yaml
from pydantic import BaseModel
import torch

PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]

class TrainConfig(BaseModel):
    precision: PRECISION_TYPES = "bfloat16"
    iterations: int = 500
    lr: float = 1e-4
    seed: int = 6666

class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    precision: PRECISION_TYPES = "float32"

class RootConfig(BaseModel):
    prompts_file: str
    train: Optional[TrainConfig]
    save: Optional[SaveConfig]

def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")

def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    return root
