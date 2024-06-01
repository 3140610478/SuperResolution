import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reload_data = False

batch_size = 1

ScalingFactor = 3

loss_weights = 1, 4

base_folder = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.abspath(os.path.join(
    base_folder, "./Networks/save/model.tar"
))
