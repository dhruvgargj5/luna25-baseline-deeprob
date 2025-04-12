import torch
from torch import nn
from torchvision import models
from torchinfo import summary


class DenseNet121(nn.Module):
    def __init__(self, densenet121_path):
        super().__init__()
        densenet121_dict = torch.load(densenet121_path, weights_only=True)
        base_model = models.densenet121(weights=None)
        base_model.load_state_dict(densenet121_dict, strict=False)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, 1)

    def forward(self, x):
        feature_map = self.features(x)
        feature_map = self.pool(feature_map)
        return self.classifier(torch.flatten(feature_map, 1))


if __name__ == "__main__":
    image = torch.randn(4, 3, 64, 64)
    model = DenseNet121("./resources/RadImageNet_pytorch/DenseNet121.pt")

    output = model(image)
    print(f"Model: {summary(model)}")
    print(f"Input: {image.shape=}")
    print(f"Output: {output.shape=}")
