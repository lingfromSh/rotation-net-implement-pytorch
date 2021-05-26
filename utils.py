import os
from typing import Tuple

import numpy as np
import torch


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


def rotation_net_accuracy(predict: torch.Tensor,
                          target: torch.Tensor,
                          matrix: np.ndarray,
                          viewpoint_num: int,
                          top_k: Tuple):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    target = target[0:-1:viewpoint_num]
    batch_size = target.shape[0]

    num_classes = predict.shape[2]
    predict = predict.cpu().numpy()
    predict = predict.transpose(1, 2, 0)
    scores = np.zeros((matrix.shape[0], num_classes, batch_size))
    output = torch.zeros((batch_size, num_classes))
    # compute scores for all the candidate poses (see Eq.(6))
    for j in range(matrix.shape[0]):
        for k in range(matrix.shape[1]):
            scores[j] = scores[j] + predict[matrix[j][k] * viewpoint_num + k]
    # for each sample #n, determine the best pose that maximizes the score (for the top class)
    for n in range(batch_size):
        j_max = int(np.argmax(scores[:, :, n]) / scores.shape[1])
        output[n] = torch.FloatTensor(scores[j_max, :, n])
    output = output.cuda()

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        res.append(correct[:k].reshape(-1).float().sum(0).mean())
    return res
