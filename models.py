import PIL.Image
import torch.nn as nn
from torchvision.models import VGG, AlexNet

from exceptions import UnsupportedBaselineModelError
from transforms import rotation_net_regulation_transform

"""
Baseline
1. alexnet
2. vgg16
"""


class Model(nn.Module):
    def __init__(self, base_model: nn.Module, categories_num: int, viewpoint_num: int):
        super(Model, self).__init__()
        num_classes = (categories_num + 1) * viewpoint_num
        if isinstance(base_model, AlexNet):
            self.features = base_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes)
            )
            self.baseline = "alexnet"
        elif isinstance(base_model, VGG):
            self.features = base_model.features
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
            self.baseline = "vgg16"
        else:
            raise UnsupportedBaselineModelError("Dont support {} as baseline model".format(base_model))

    def forward(self, x):
        x = self.features(x)
        if self.baseline == "alexnet":
            x = x.view(x.shape[0], 256 * 6 * 6)
        elif self.baseline == 'vgg16':
            x = x.view(x.size(0), -1)
        return self.classifier(x)

    def predict(self, image: PIL.Image):
        self.eval()
        tensor = rotation_net_regulation_transform(image)
        y = self.forward(tensor)
        return y
