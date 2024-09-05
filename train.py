from build_voc_dataset import read_annotation_with_class_and_position, read_image, build_label_by_one_data
import os
from read_voc import read_image, class_balance
import numpy as n
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms 
# from yolo import YOLOv1
from yolo_resnet34_7x7 import YOLOv1
from loss_of_torch import yolo_loss_sigmoid


transform = transforms.Compose([  
    transforms.ToTensor(),         # 将 PIL Image 或 ndarray 转换为 tensor，并归一化到 [0.0, 1.0]  
])

class CustomImageDataset(Dataset):  
    def __init__(self, dir_path):  
        self.dir_path = dir_path  
        self.image_files = [f.replace(".jpg", '') for f in os.listdir(dir_path) if f.endswith(('.jpg'))]  
  
    def __len__(self):  
        return len(self.image_files)  
    
    def __getitem__(self, idx):  
        file = self.image_files[idx]  
        y = build_label_by_one_data(read_annotation_with_class_and_position(os.path.join(self.dir_path, file+".xml")))
        y = torch.Tensor(y).float()
        x = read_image(os.path.join(self.dir_path, file+".jpg"))
        x = transform(x)
        return x, y  


def getXY_by_numbers(dir_path, files):
    xs = []
    ys = []
    for i, file in enumerate(list(files)):
        # print(file)
        one_data = read_annotation_with_class_and_position(os.path.join(dir_path, file+".xml"))
        print(one_data)
        y = build_label_by_one_data(one_data)
        x = read_image(os.path.join(dir_path, file+".jpg"))
        xs.append(x[n.newaxis, ...])
        ys.append(y[n.newaxis, ...])
    xs = n.concatenate(xs, axis=0)
    xs = xs / 255.
    xs = n.transpose(xs, (0, 3, 1, 2))
    ys = n.concatenate(ys, axis=0)
    print(xs.shape, xs.max(), xs.min())
    print(ys.shape, ys.max(), ys.min())
    print(n.where(ys > 1.0))
    return xs, ys, files

def getXY(dir_path):
    files = os.listdir(dir_path)
    files = [i[:-4] for i in files]
    files = set(files)
    return getXY_by_numbers(dir_path, ['705171'])
    # return getXY_by_numbers(dir_path, ['705171'])
    # return getXY_by_numbers(dir_path, files)

def train():
    # torch.manual_seed(0)
    # n.random.seed(0)
    model_path = "model/yolov1.pt"
    opt_path = "model/optim.pt"
    epochs_path = 'model/epochs.pt'
    dir_path = './aug'
    device = "cuda:0"  if torch.cuda.is_available() else "cpu"
    # 定义批量大小
    batch_size = 8
    model = YOLOv1(True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    last_epochs = 0
    # ..................................................................................................

    # xs, ys, _ = getXY(dir_path)
    # xs_tensor = torch.tensor(xs, dtype=torch.float32)
    # ys_tensor = torch.tensor(ys, dtype=torch.float32)
    # dataset = TensorDataset(xs_tensor, ys_tensor)

    dataset = CustomImageDataset(dir_path)


    # ..................................................................................................

    model.load(model_path)
    opt_dict = torch.load(opt_path)
    # opt_dict['lr'] = 1e-3
    optimizer.load_state_dict(opt_dict)
    last_epochs = torch.load(epochs_path)

    # ..................................................................................................

    # 创建数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("Last epochs: {:.3f}".format(last_epochs))

    balance_weight = class_balance(dir_path)
    balance_weight = torch.tensor(balance_weight, device=device, dtype=torch.float32)
    print(balance_weight)
    
    # 训练网络
    num_epochs = 1000
    si = 0
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader, 0):
            print('inputs labels', inputs.max(), inputs.min(), labels.max(), labels.min())
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, obj_conf_loss, obj_xy_loss,obj_wh_loss, obj_class_loss, noobj_conf_loss = yolo_loss_sigmoid(outputs, labels, balance_weight)
            loss.backward()
            optimizer.step()
            if si % 1 == 0:
                print('[{}/{}, {}/{}] loss: {:.5f}   c {:.5f} nc {:.5f} xy {:.5f} wh {:.5f} cl {:.5f}'.format(num_epochs, epoch + 1, len(train_loader), i + 1, loss.item(), obj_conf_loss.item(), noobj_conf_loss.item(), obj_xy_loss.item(), obj_wh_loss.item(), obj_class_loss.item()))
            if si % 500 == 0:
                print("save model")
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), opt_path)
                torch.save(last_epochs + epoch + (i+1) / len(train_loader), epochs_path)
            si += 1
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), opt_path)
    torch.save(last_epochs + epoch, epochs_path)



if __name__ == "__main__":
    train()
















