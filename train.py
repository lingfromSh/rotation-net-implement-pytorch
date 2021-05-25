import math
import os
import time
from typing import Dict

import numpy as np
import torch
from operator import sub
from PyInquirer import prompt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import alexnet, vgg16

from commands import Command
from exceptions import UnsupportedBaselineModelError
from models import Model
from transforms import rotation_net_regulation_transformer
from utils import AverageMeter, adjust_learning_rate, random_input
from test import TestCommand


class TrainCommand(Command):
    name = "Training"

    def command(self, context: Dict):
        baseline_menu = {
            "type": "input",
            "name": "baseline",
            "message": "Baseline Model:",
            "default": "alexnet",
        }
        baseline = prompt(baseline_menu)["baseline"]

        pretrained_menu = {
            "type": "confirm",
            "name": "pretrained",
            "message": "Use pretrained model?",
            "default": True,
        }
        pretrained = prompt(pretrained_menu)["pretrained"]

        resume_menu = {
            "type": "confirm",
            "name": "resume",
            "message": "Resume?",
            "default": True,
        }
        resume = prompt(resume_menu)["resume"]

        learning_rate_menu = {
            "type": "input",
            "name": "lr",
            "message": "Learning rate:",
            "default": "0.01",
        }
        lr = float(prompt(learning_rate_menu)["lr"])

        epoch_menu = {
            "type": "input",
            "name": "epoch",
            "message": "Epoch:",
            "default": "1500",
        }
        epoch = int(prompt(epoch_menu)["epoch"])

        category_menu = {
            "type": "input",
            "name": "categories_num",
            "message": "Category num:",
            "default": "40",
        }
        categories_num = int(prompt(category_menu)["categories_num"])

        viewpoint_menu = {
            "type": "input",
            "name": "viewpoint_num",
            "message": "Viewpoint num:",
            "default": "12",
        }
        viewpoint_num = int(prompt(viewpoint_menu)["viewpoint_num"])

        batch_size_menu = {
            "type": "input",
            "name": "batch_size",
            "message": "Batch Size:",
            "default": "240",
        }
        batch_size = int(prompt(batch_size_menu)["batch_size"])

        print_freq_menu = {
            "type": "list",
            "name": "print_freq",
            "message": "Print frequency",
            "choices": ["10", "20", "100", "200"],
            "default": "100",
        }
        print_freq = int(prompt(print_freq_menu)["print_freq"])

        save_freq_menu = {
            "type": "list",
            "name": "save_freq",
            "message": "Save checkpoint frequency",
            "choices": ["1", "5", "10", "20", "50", "100", "200"],
            "default": "5",
        }
        save_freq = int(prompt(save_freq_menu)["save_freq"])

        print(
            "",
            " ============= Training parameters  =============",
            f"Baseline Model: {baseline}",
            f"Pretrained: {pretrained}",
            f"Resume: {resume}",
            f"Print frequency: {print_freq}",
            f"Save checkpoint frequency: {save_freq}",
            f"Learning rate: {lr}",
            f"Epoch: {epoch}",
            f"Categories Num: {categories_num}",
            f"Viewpoint Num: {viewpoint_num}",
            f"Batch Size: {batch_size}",
            sep="\n",
        )

        self.train(
            baseline,
            pretrained,
            resume,
            print_freq,
            save_freq,
            lr,
            epoch,
            categories_num,
            viewpoint_num,
            batch_size,
        )

    @classmethod
    def train(
        cls,
        baseline,
        pretrained,
        resume,
        print_freq,
        save_freq,
        lr,
        epoch_amount,
        categories_num,
        viewpoint_num,
        batch_size,
    ):
        if baseline == "alexnet":
            baseline_model = alexnet(pretrained=pretrained)
        elif baseline == "vgg16":
            baseline_model = vgg16(pretrained=pretrained)
        else:
            raise UnsupportedBaselineModelError

        train_dataset = DataLoader(
            ImageFolder(
                "data/ModelNet40v2/modelnet/train",
                transform=rotation_net_regulation_transformer,
            ),
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
        )
        if resume and os.path.exists("rotation_net.pth"):
            pth = torch.load("rotation_net.pth")
            model = pth["model"]
            epoch = pth["epoch"] + 1
            optimizer = pth["optimizer"]
        else:
            epoch = 1
            model = Model(
                baseline_model,
                categories_num=categories_num,
                viewpoint_num=viewpoint_num,
            ).cuda()
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
                momentum=0.9,
                weight_decay=1e-4,
            )

        criterion = torch.nn.CrossEntropyLoss().cuda()
        matrix = np.load("matrix2.npy")

        total_batch = math.ceil(len(train_dataset.dataset) / batch_size)
        losses = AverageMeter()
        timer = AverageMeter()

        model.train()
        for epoch in range(epoch, epoch_amount + 1):

            print(f"============= Epoch {epoch} =============")
            adjust_learning_rate(optimizer, epoch, lr)
            random_input(train_dataset, viewpoint_num)

            for batch, (samples, targets) in enumerate(train_dataset, start=1):
                start = time.time()

                # (batch_size, (categories_num + 1) * viewpoint_num)
                output = model(samples.cuda())
                targets = targets.cuda()
                # (batch_size*viewpoint_num, categories_num + 1)
                object_num = targets.shape[0] // viewpoint_num

                output = output.view(-1, categories_num + 1)
                output_possibility = torch.log_softmax(output, dim=1)
                output_possibility = sub(
                    output_possibility[:, :-1],
                    torch.t(
                        output_possibility[:, -1]
                        .repeat(1, output_possibility.shape[1] - 1)
                        .view(output_possibility.shape[1] - 1, -1)
                    ),
                )

                final_possibility = (
                    output_possibility.view(
                        -1, viewpoint_num * viewpoint_num, categories_num
                    )
                    .data.cpu()
                    .numpy()
                )
                final_possibility = final_possibility.transpose(1, 2, 0)

                scores = torch.zeros(matrix.shape[0], categories_num, object_num).cuda()
                for i in range(matrix.shape[0]):
                    for j in range(matrix.shape[1]):
                        scores[i] = (
                            scores[i]
                            + final_possibility[matrix[i][j] * viewpoint_num + j]
                        )

                labels = torch.full((viewpoint_num * viewpoint_num * object_num, ), fill_value=categories_num)

                for i in range(object_num):
                    j_max = np.argmax(scores[:, targets[i * viewpoint_num], i])
                    for j in range(matrix.shape[1]):
                        labels[
                            i * viewpoint_num * viewpoint_num
                            + matrix[j_max][j] * viewpoint_num
                            + j
                        ] = targets[i * viewpoint_num]

                loss = criterion(output, labels.cuda())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.update(loss.item(), batch_size)
                timer.update(time.time() - start, 1)

                if batch % print_freq == 0:
                    print(
                        f"[Batch {batch}/{total_batch}]:"
                        f"\nTime - {timer.avg}"
                        f"\nAverage Loss - {losses.avg}"
                        f"\nCur Batch Loss - {losses.val}"
                    )

            if epoch % save_freq == 0:
                torch.save(
                    {"model": model, "epoch": epoch, "lr": lr, "optimizer": optimizer},
                    f="rotation_net.pth",
                )

                TestCommand.test(batch_size, viewpoint_num)
