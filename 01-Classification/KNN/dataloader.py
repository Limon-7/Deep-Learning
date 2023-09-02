import os
from pathlib import Path
import torch
from torchvision.datasets import CIFAR10


ROOT_DIR = Path(__file__).parent.resolve()
IMAGE_DIR = ROOT_DIR / "data"
DIR_PATH = Path(ROOT_DIR / "data" / "cifar-10-batches-py")


def _extract_tensors(dset, num=None):
    x = torch.tensor(dset.data, dtype=torch.float32).permute(0, 3, 1, 2).div_(255)
    y = torch.tensor(dset.targets, dtype=torch.int64)
    if num is not None:
        if num <= 0 or num > x.shape[0]:
            raise ValueError(
                "Invalid value num=%d; must be in the range [0, %d]" % (num, x.shape[0])
            )
        x = x[:num].clone()
        y = y[:num].clone()
    return x, y


def cifar10(num_train=None, num_test=None):
    download = not DIR_PATH.exists()
    trainset = CIFAR10(root=IMAGE_DIR, download=download, train=True)
    testset = CIFAR10(root=IMAGE_DIR, train=False)
    x_train, y_train = _extract_tensors(trainset, trainset.data.shape[0])
    x_test, y_test = _extract_tensors(testset, testset.data.shape[0])

    return x_train, y_train, x_test, y_test
