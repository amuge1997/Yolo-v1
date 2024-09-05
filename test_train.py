from build_voc_dataset import read_annotation_with_class_and_position, build_label_by_one_data, label_back_to_one_data
import os
from read_voc import read_image
import numpy as n
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from yolo import YOLOv1
from yolo_resnet34_7x7 import YOLOv1
from loss_of_torch import yolo_loss_sigmoid
from global_param import grid_number
from train import getXY_by_numbers
from show_voc import show_image_with_bboxs, show_image_with_bboxs_and_names


def read_image_and_process_as_one_batch(file):
    x = read_image(file)
    x = x / 255
    x = x[n.newaxis, ...]
    x = n.transpose(x, (0, 3, 1, 2))
    return x


def test():
    data_dir = './aug'
    path = "model/yolov1.pt"

    image_number = '122393'
    image_number = '100422'
    image_number = '106416'
    # image_number = '107213'
    image_number = '718486'
    # image_number = '718881'
    image_number = '719206'
    # image_number = '720373'
    # image_number = '720413'
    
    cls_th = 0.0
    th = 0.25


    model = YOLOv1(True)
    model.load(path)
    model.eval()
    xs = read_image_and_process_as_one_batch("{}/{}.jpg".format(data_dir,image_number))
    xs_tensor = torch.tensor(xs, dtype=torch.float32)
    ys_tensor = model(xs_tensor)
    print(ys_tensor.shape)
    y0 = ys_tensor.detach().numpy()[0]
    print("\n"*3)

    one_label = n.zeros((grid_number, grid_number, 25))

    max_yx = {"y":None,"x":None}
    max_label = n.zeros((25))

    for yi in range(grid_number):
        for xi in range(grid_number):
            grid = y0[yi, xi]
            if grid[0] > th and grid[0] >= grid[1]:
                one_label[yi, xi, 0] = grid[0]
                one_label[yi, xi, 1:5] = grid[2:6]
                one_label[yi, xi, 5:] = grid[10:]
                if grid[0] > max_label[0]:
                    max_label[0] = grid[0]
                    max_label[1:5] = grid[2:6]
                    max_label[5:] = grid[10:]
                    max_yx['y'] = yi
                    max_yx['x'] = xi
            elif grid[1] > th and grid[1] >= grid[0]:
                one_label[yi, xi, 0] = grid[1]
                one_label[yi, xi, 1:5] = grid[6:10]
                one_label[yi, xi, 5:] = grid[10:]
                if grid[1] > max_label[0]:
                    max_label[0] = grid[1]
                    max_label[1:5] = grid[6:10]
                    max_label[5:] = grid[10:]
                    max_yx['y'] = yi
                    max_yx['x'] = xi
    
    print("\n"*3)
    print(y0[:, :, :2].max())
    print("\n"*3)
    print(one_label.shape)
    one_data = label_back_to_one_data(one_label, cls_th=cls_th)
    print(len(one_data['objs']))
    print(one_data)
    image = read_image("{}/{}.jpg".format(data_dir,image_number))
    # show_image_with_bboxs(image, [obj['box'] for obj in one_data['objs']])
    show_image_with_bboxs_and_names(image, one_data['objs'])
    print("\n"*3)

    max_one_label = n.zeros((grid_number, grid_number, 25))
    if max_yx['y'] is not None:
        max_one_label[max_yx['y'], max_yx['x'], :] = max_label
    max_one_data = label_back_to_one_data(max_one_label, cls_th=cls_th)
    print(max_yx, max_label)
    print(max_one_data)
    print("max conf: ", max_label[0])
    image = read_image("{}/{}.jpg".format(data_dir,image_number))
    # show_image_with_bboxs(image, [obj['box'] for obj in max_one_data['objs']])
    show_image_with_bboxs_and_names(image, max_one_data['objs'])
    print("\n"*3)

    xs, ys, files = getXY_by_numbers(data_dir, [image_number])
    print(n.where(ys[:, :, :, 0] >= 1.0, 1, 0))
    y0s = ys[files.index(image_number)]
    one_data = label_back_to_one_data(y0s)
    image = read_image("{}/{}.jpg".format(data_dir,image_number))
    # show_image_with_bboxs(image, [obj['box'] for obj in one_data['objs']])
    show_image_with_bboxs_and_names(image, one_data['objs'])



if __name__ == "__main__":
    test()









