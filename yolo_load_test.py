import torch
import torch.nn as nn
import torchvision.models as models
class YOLOv1(nn.Module):
    def __init__(self, is_pretrained=False):
        super().__init__()
        vgg16 = models.vgg16(pretrained=is_pretrained)
        vgg16.eval()
        for param in vgg16.parameters():  
            param.requires_grad = False 
        features = nn.Sequential(*list(vgg16.children())[0])

        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1470)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        for name, param in self.named_parameters():
            if name.startswith("features"):
                param.requires_grad = False 

image = torch.ones(size=(1, 3, 224, 224))

yolo = YOLOv1()

yolo.load("pt/yolo.pt")
yolo.eval()

y = yolo(image)

print(y.shape)
print(y[0, 0:10])













