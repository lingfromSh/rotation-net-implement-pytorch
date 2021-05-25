import os

import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import BatchSampler


class AverageMeter:
    def __init__(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.total = 0  # 总体值
        self.amount = 0  # 总体计数

    def update(self, val, count):
        self.val = val
        self.total += val * count
        self.amount += count
        self.avg = self.total / self.amount

    @property
    def average(self):
        return self.avg


def get_loss_weight(classes_num, viewpoint_num, samples):
    count = np.zeros(((classes_num + 1), viewpoint_num))
    for sample, label in samples:
        count[label] += 1
    median = np.median(count)
    count[classes_num] = median
    count = median / count
    return torch.from_numpy(count).float()


def get_categories(dataset_path):
    categories = {}
    for idx, name in enumerate(sorted(os.listdir(dataset_path))):
        categories[idx] = name
    return categories


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 200 epochs"""
    lr = lr * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        print("Learning Rate: {lr:.6f}".format(lr=param_group["lr"]))


def random_input(dataset, viewpoint_num):
    sorted_images = sorted(dataset.dataset.imgs)
    total = len(dataset.dataset)
    object_num = total // viewpoint_num

    idxes = np.ndarray((viewpoint_num, object_num), dtype=np.int64)
    idxes[0] = np.random.permutation(range(object_num)) * viewpoint_num

    for i in range(1, viewpoint_num):
        idxes[i] = idxes[0] + i

    sorted_images = [
        sorted_images[i] for i in idxes.T.reshape(viewpoint_num * object_num)
    ]
    dataset.dataset.imgs = sorted_images
    dataset.dataset.samples = dataset.dataset.imgs


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from transforms import rotation_net_regulation_transformer

    train_dataset = DataLoader(
        ImageFolder(
            "data/ModelNet40v1/modelnet/train",
            transform=rotation_net_regulation_transformer,
        ),
        batch_size=120,
        shuffle=False,
        drop_last=True,
    )
    for samples, targets in train_dataset:
        print(samples, targets)
    print(train_dataset.dataset.imgs[0:12])
    random_input(train_dataset, 12)
    print(train_dataset.dataset.imgs[0:12])
