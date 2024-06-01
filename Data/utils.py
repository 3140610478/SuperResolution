import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import center_crop, to_pil_image

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


def join_path(*args):
    return os.path.abspath(os.path.join(*args))


class ToDevice(torch.nn.Module):
    def __init__(self, device=config.device):
        super().__init__()
        self.device = device

    def forward(self, input: torch.Tensor):
        return input.to(self.device)


class DivisibleCrop(torch.nn.Module):
    def __init__(self, factor=4):
        super().__init__()
        self.factor = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        size = torch.tensor(input.shape[-2:])
        size = size - size % self.factor
        output = center_crop(input, size.tolist())
        return output