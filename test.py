import os
from typing import Dict

import numpy as np
import torch
from PyInquirer import prompt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from commands import Command
from transforms import rotation_net_regulation_transformer
from utils import AverageMeter, random_input


class TestCommand(Command):
    name = "Testing"

    def command(self, context: Dict):
        batch_size_menu = {
            "type": "input",
            "name": "batch_size",
            "message": "Batch Size:",
            "default": "240",
        }
        batch_size = int(prompt(batch_size_menu)["batch_size"])

        viewpoint_menu = {
            "type": "input",
            "name": "viewpoint_num",
            "message": "Viewpoint num:",
            "default": "80",
        }
        viewpoint_num = int(prompt(viewpoint_menu)["viewpoint_num"])

        self.test(batch_size, viewpoint_num)

    @classmethod
    def test(cls, batch_size, viewpoint_num):
        if not os.path.exists("rotation_net.pth"):
            print("模型未找到")

        pth = torch.load("rotation_net.pth")
        model = pth["model"]
        epoch = pth["epoch"]

        test_dataset = DataLoader(
            ImageFolder(
                "data/ModelNet40v2/modelnet/test", 
                transform=rotation_net_regulation_transformer
            ),
            pin_memory=True,
            batch_size=batch_size,
            shuffle=False,
        )
        random_input(test_dataset, viewpoint_num)
        matrix = np.load("matrix2.npy")

        def my_accuracy(output_, target, topk=(1,)):
            """Computes the precision@k for the specified values of k"""
            maxk = max(topk)
            target = target[0:-1:viewpoint_num]
            batch_size = target.size(0)

            num_classes = output_.size(2)
            output_ = output_.cpu().numpy()
            output_ = output_.transpose(1, 2, 0)
            scores = np.zeros((matrix.shape[0], num_classes, batch_size))
            output = torch.zeros((batch_size, num_classes))
            # compute scores for all the candidate poses (see Eq.(6))
            for j in range(matrix.shape[0]):
                for k in range(matrix.shape[1]):
                    scores[j] = scores[j] + output_[matrix[j][k] * viewpoint_num + k]
            # for each sample #n, determine the best pose that maximizes the score (for the top class)
            for n in range(batch_size):
                j_max = int(np.argmax(scores[:, :, n]) / scores.shape[1])
                output[n] = torch.FloatTensor(scores[j_max, :, n])
            output = output.cuda()

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.contiguous().view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

        print(f"模型已经训练: {epoch}次")
        model.eval()
        top1 = AverageMeter()
        top5 = AverageMeter()

        with torch.no_grad():

            for i, (samples, targets) in enumerate(test_dataset):
                targets = targets.cuda()
                # compute output
                output = model(samples.cuda())
                # log_softmax and reshape output
                num_classes = int(output.size(1) / viewpoint_num) - 1
                output = output.view(-1, num_classes + 1)
                output = torch.log_softmax(output, dim=1)
                output = output[:, :-1] - torch.t(
                    output[:, -1].repeat(1, output.size(1) - 1).view(output.size(1) - 1, -1))
                output = output.view(-1, viewpoint_num * viewpoint_num, num_classes)

                # measure accuracy and record loss
                prec1, prec5 = my_accuracy(output.data, targets, topk=(1, 5))
                top1.update(prec1.item(), samples.shape[0] / viewpoint_num)
                top5.update(prec5.item(), samples.shape[0] / viewpoint_num)

        print(f"Top1 Accuracy: {top1.avg}")
        print(f"Top5 Accuracy: {top5.avg}")
