import torch
import torch.nn as nn
import torchvision.models as models
from global_param import grid_number


class YOLOv1(nn.Module):
    def __init__(self, is_pretrained=False):
        super().__init__()

        base_channels = 1
        features = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 112

            nn.Conv2d(base_channels, base_channels*2**1, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 56

            nn.Conv2d(base_channels*2**1, base_channels*2**2, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 28

            nn.Conv2d(base_channels*2**2, base_channels*2**3, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 14

            nn.Conv2d(base_channels*2**3, base_channels*2**4, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 7
        )

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(base_channels*2**4 * grid_number * grid_number, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, grid_number * grid_number * 30),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        x = x.view(x.shape[0], grid_number, grid_number, 30)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        for name, param in self.named_parameters():
            if name.startswith("features"):
                param.requires_grad = False 

if __name__ == "__main__":
    image = torch.ones(size=(1, 3, 224, 224))
    # y = features(image)
    # print(y.shape)

    yolo = YOLOv1(True)
    yolo.eval()

    y = yolo(image)
    print(y.shape)
    print(y[0, 0:10])

    # yolo.save("pt/yolo.pt")













