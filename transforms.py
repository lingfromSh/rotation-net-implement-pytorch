from torchvision import transforms

normalize = transforms.Normalize(
    std=[0.229, 0.224, 0.225],
    mean=[0.485, 0.456, 0.406]
)

rotation_net_regulation_transformer = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def rotation_net_regulation_transform(x):
    return rotation_net_regulation_transformer(x)
