import os
from typing import Dict

import numpy as np
import torch
from PyInquirer import prompt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from commands import Command
from transforms import rotation_net_regulation_transformer
from utils import AverageMeter, random_input, rotation_net_accuracy


class ValidateCommand(Command):
    name = "Validating"

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

        self.validate(batch_size, viewpoint_num)

    @classmethod
    def validate(cls, batch_size, viewpoint_num):
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
                output = output[:, :-1] - torch.t(output[:, -1].repeat(1, output.shape[1] - 1).view(output.shape[1] - 1, -1))
                output = output.view(-1, viewpoint_num *
                                     viewpoint_num, num_classes)

                # measure accuracy and record loss
                pred_1, pred_5 = rotation_net_accuracy(output.data,
                                                       targets,
                                                       matrix=matrix,
                                                       viewpoint_num=viewpoint_num,
                                                       top_k=(1, 5))
                top1.update(pred_1.item(), samples.shape[0] / viewpoint_num)
                top5.update(pred_5.item(), samples.shape[0] / viewpoint_num)

        print(f"Top1 Accuracy: {top1.avg}")
        print(f"Top5 Accuracy: {top5.avg}")
