import os
import sys
import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots

base_folder = os.path.dirname(os.path.abspath(__file__))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from Networks.PixelShuffler import PixelShuffler as PS
    from Data import BSDS
    import config


@torch.no_grad()
def predict(model, data, eps=1e-8):
    data_loader = data.test_loader
    output_folder = os.path.abspath(os.path.join(
        base_folder, "./Data/predicted",
    ))
    os.makedirs(output_folder, exist_ok=True)
    for i, sample in enumerate(tqdm(data_loader)):
        x, y = sample
        x, y = x.to(config.device), y.to(config.device)

        h = model(x)
        x, y, h = x.squeeze(0), y.squeeze(0), h.squeeze(0)

        h = torch.max(
            torch.min(
                h,
                torch.ones_like(h),
            ),
            torch.zeros_like(h),
        )

        x, y, h = (to_pil_image(i.squeeze(0)) for i in (x, y, h))
        x = x.save(os.path.abspath(os.path.join(
            output_folder, f"./{i}_orig.jpg"
        )))
        y = y.save(os.path.abspath(os.path.join(
            output_folder, f"./{i}_gt.jpg"
        )))
        h = h.save(os.path.abspath(os.path.join(
            output_folder, f"./{i}_pred.jpg"
        )))
        pass


if __name__ == "__main__":
    checkpoint = torch.load(config.save_path)
    ps = PS(3, config.ScalingFactor)
    ps.load_state_dict(checkpoint["state_dict"])
    predict(ps.to(config.device), BSDS)
    pass
