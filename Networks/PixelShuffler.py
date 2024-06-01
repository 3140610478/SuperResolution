import os
import sys
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config


class PixelShuffler(nn.Sequential):
    def __init__(self, in_channels=3, factor=config.ScalingFactor) -> None:
        super().__init__(
            nn.Conv2d(in_channels, 64, 5, 1, 2),
            nn.Tanh(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Tanh(),
            nn.Conv2d(32, factor**2 * 3, 3, 1, 1),
            nn.PixelShuffle(config.ScalingFactor),
        )


if __name__ == "__main__":
    ps = PixelShuffler(3, config.ScalingFactor).to("cuda")
    print(str(ps))
    a = torch.zeros((16, 3, 120, 90)).to("cuda")
    while True:
        print(a.shape, ps.forward(a).shape)
    pass
