import os
import sys
import pickle
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor, gaussian_blur
from PIL import Image


base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Data.utils import join_path, ToDevice, DivisibleCrop
    from Log.Logger import getLogger


# The file structure was reorganized as follows:
#     SuperResolution/Data/:
#         original:
#             bsds500:
#                 trainval:
#                     xxx.jpg
#                     xxx.jpg
#                     .
#                     .
#                     .
#                     xxx.jpg
#                 test:
#                     xxx.jpg
#                     xxx.jpg
#                     .
#                     .
#                     .
#                     xxx.jpg
#                 ReadMe.txt
#             bsds500.zip
#         preprocessed:
#             train.pickle    {"inputs": List[torch.Tensor] of RGB images, "targets": List[torch.Tensor] in one_hot encoding}
#             val.pickle      {"inputs": List[torch.Tensor] of RGB images, "targets": List[torch.Tensor] in one_hot encoding}
#             test.pickle     {"inputs": List[torch.Tensor] of RGB images, "targets": List[torch.Tensor] in one_hot encoding}
#         predicted:
#             xxx_orig.jpg
#             xxx_gt.jpg
#             xxx_pred.jpg
#             .
#             .
#             .



original_folder = join_path(base_folder, "./Data/original/bsds500")
preprocessed_folder = join_path(base_folder, "./Data/preprocessed")
os.makedirs(preprocessed_folder, exist_ok=True)

train_path = join_path(preprocessed_folder, "./train.pickle")
val_path = join_path(preprocessed_folder, "./val.pickle")
test_path = join_path(preprocessed_folder, "./test.pickle")


split_point = 0.9
train = os.listdir(join_path(original_folder, "./trainval"))
train = [i for i in train if i.endswith(".jpg")]
train = [join_path(original_folder, f"./trainval/{i}") for i in train]
random.shuffle(train)
split_point = int(len(train) * split_point)
train, val = train[:split_point], train[split_point:]
test = os.listdir(join_path(original_folder, "./test"))
test = [i for i in test if i.endswith(".jpg")]
test = [join_path(original_folder, f"./test/{i}") for i in test]


def _preprocess(files: list[str], output_path: str):
    crop = DivisibleCrop(config.ScalingFactor)
    inputs, targets = [], []

    print(f"Processing {output_path}")
    for file in tqdm(files):
        target = to_tensor(Image.open(file))
        target = target.to(config.device)
        target = crop(target)

        input_shape = [i // config.ScalingFactor for i in target.shape[-2:]]
        input = target.unsqueeze(0)
        input = gaussian_blur(input, 5)
        input = F.interpolate(
            input, input_shape, mode='nearest',
        ).squeeze(0)

        inputs.append(input.cpu())
        targets.append(target.cpu())

    with open(output_path, "wb") as f:
        pickle.dump({"inputs": inputs, "targets": targets}, f)


class _BSDS_Dataset(Dataset):
    def __init__(self, path, transform=ToDevice(config.device)):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.inputs, self.targets = data["inputs"], data["targets"]
        self.LEN = len(self.inputs)
        for i in range(self.LEN):
            self.inputs[i].requires_grad_(False)
            self.targets[i].requires_grad_(False)
        self.transform = transform

    def __len__(self):
        return self.LEN

    def __getitem__(self, index):
        input, target = self.inputs[index].clone(), self.targets[index].clone()
        input, target = self.transform(input), self.transform(target)
        return input, target


try:
    if config.reload_data or not os.path.exists(preprocessed_folder):
        raise Exception()
    train_set = _BSDS_Dataset(train_path)
    val_set = _BSDS_Dataset(val_path)
    test_set = _BSDS_Dataset(test_path)
except:
    _preprocess(train, train_path)
    _preprocess(val, val_path)
    _preprocess(test, test_path)
    train_set = _BSDS_Dataset(train_path)
    val_set = _BSDS_Dataset(val_path)
    test_set = _BSDS_Dataset(test_path)
len_train, len_val, len_test = \
    len(train_set), len(val_set), len(test_set)

train_loader = DataLoader(
    train_set, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=config.batch_size, shuffle=True)
test_loader = DataLoader(
    test_set, batch_size=config.batch_size, shuffle=True)

if __name__ == "__main__":
    for x, y in train_loader:
        print(x.shape, y.shape)
        print(x.min(), x.max(), y.min(), y.max())
