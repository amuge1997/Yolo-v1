import torch
import torch.nn as nn
import torchvision.models as models
from global_param import grid_number, image_size
import torch.nn.functional as F


class YOLOv1(nn.Module):
    def __init__(self, is_pretrained=False):
        super().__init__()
        pre_model = models.resnet34(pretrained=is_pretrained)
        pre_model.eval()
        for param in pre_model.parameters():  
            param.requires_grad = False 
        # print(pre_model)
        # exit()
        features = nn.Sequential(*list(pre_model.children())[:-2])

        # self.neck = nn.Sequential(
        #     nn.Conv2d(512, 512, 2, 2, 0),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(inplace=True)
        # )

        self.features = features
        # self.classifier = nn.Sequential(
        #     nn.Conv2d(512, 512, 2, 2, 0),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(512, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(256, 512, grid_number, 1, 0),
        #     nn.BatchNorm2d(512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.UpsamplingNearest2d(scale_factor=2),
        #     nn.Flatten(),
        #     nn.Linear(256 * grid_number ** 2, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(1024, 11 * grid_number ** 2),      # batch, 5000+
        #     nn.BatchNorm1d(11 * grid_number ** 2),
        # )
        self.classifier1 = nn.Sequential(
            # nn.Conv2d(512, 256, 2, 2, 0),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),                              # 512, 7, 7
            nn.AvgPool2d(2),                        # 512, 7, 7
            
        )
        self.classifier2 = nn.Sequential(
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),                             # 256, 7, 7
        )
        self.classifier3 = nn.Sequential(
            
            nn.Conv2d(512, 512, grid_number, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),                             # 256, 1, 1
        )
        self.classifier4 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, grid_number, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(),                             # 256, 7, 7
        )
        self.classifier5 = nn.Sequential(
            nn.Conv2d(512, 30, 1, 1, 0),
            # nn.BatchNorm2d(11),                                     # 11, 7, 7
        )
    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.shape[0], -1)
        # print(x.shape)
        # x = self.neck(x)
        # print(x.shape)
        # exit()
        # print('f ', x.max(), x.min(), x.dtype)
        x = self.classifier1(x)
        # print('c1 ', x.max(), x.min(), x.dtype)
        x = self.classifier2(x)
        # print('c2 ', x.max(), x.min(), x.dtype)
        x = self.classifier3(x)
        # print('c3 ', x.max(), x.min(), x.dtype)
        x = self.classifier4(x)
        # print('c4 ', x.max(), x.min(), x.dtype)
        x = self.classifier5(x)
        # print('c5 ', x.max(), x.min(), x.dtype)
        # x = x.view(x.shape[0], grid_number, grid_number, 11)
        # print(x.shape)
        x = x.permute((0, 2, 3, 1))
        # print(x.shape)
        # x[:, :, :, 0:2] = F.sigmoid(x[:, :, :, 0:2])        # c1, c2
        # x[:, :, :, 2:4] = F.sigmoid(x[:, :, :, 2:4])        # x1, y1
        # x[:, :, :, 6:8] = F.sigmoid(x[:, :, :, 6:8])        # x2, y2
        # x[:, :, :, 4:6] = F.sigmoid(x[:, :, :, 4:6])*1      # not wh1
        # x[:, :, :, 8:10] = F.sigmoid(x[:, :, :, 8:10])*1    # not wh2
        # x[:, :, :, 10:] = F.sigmoid(x[:, :, :, 10:])        # class
        x = F.sigmoid(x)
        # print('s ', x.max(), x.min())
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        for name, param in self.named_parameters():
            if name.startswith("features"):
                param.requires_grad = False 

if __name__ == "__main__":
    image = torch.ones(size=(1, 3, image_size, image_size))
    # y = features(image)
    # print(y.shape)
    
    yolo = YOLOv1(True)
    yolo.eval()

    y = yolo(image)
    print(y.shape)

    # yolo.save("pt/yolo.pt")

    # torch.save(models.resnet34().state_dict(), "pt/resnet34.pt")










