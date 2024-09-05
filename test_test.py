from build_voc_dataset import label_back_to_one_data
from read_voc import read_image
import numpy as n
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
# from yolo import YOLOv1
from yolo_resnet34_7x7 import YOLOv1
from loss_of_torch import yolo_loss_sigmoid
from global_param import grid_number, image_size
from train import getXY_by_numbers
from show_voc import show_image_with_bboxs, show_image_with_bboxs_and_names
from PIL import Image 
import cv2


def pad_and_resize_image(image_path, target_size):
    # Open image
    image = Image.open(image_path)
    
    # Get original image size
    original_width, original_height = image.size
    
    # Calculate padding dimensions
    max_dimension = max(original_width, original_height)
    pad_width = max_dimension - original_width
    pad_height = max_dimension - original_height
    
    # Calculate padding values
    left = pad_width // 2
    top = pad_height // 2
    right = pad_width - left
    bottom = pad_height - top
    
    # Pad image
    padded_image = Image.new(image.mode, (max_dimension, max_dimension), color='black')
    padded_image.paste(image, (left, top))
    
    # Resize image
    resized_image = padded_image.resize((target_size, target_size))
    
    return resized_image



def process_as_one_batch(x):
    x = x / 255
    x = x[n.newaxis, ...]
    x = n.transpose(x, (0, 3, 1, 2))
    return x


def image_channels_to_cv_channels(x):
    r = x[:, :, 0:1]
    g = x[:, :, 1:2]
    b = x[:, :, 2:3]
    x = n.concatenate((b, g, r), axis=2)
    return x


def read_image_pad_and_resize_and_process(image_path, target_size):  
    image = pad_and_resize_image(image_path, target_size)
    x = n.array(image)
    x = image_channels_to_cv_channels(x)
    src_pad_and_resize_image = x
    image = Image.fromarray(x)
    # image.show()
    x = process_as_one_batch(x)
    return x, src_pad_and_resize_image


def test():

    image_number = '6'

    data_dir = './test_image'
    path = "model/yolov1.pt"

    cls_th = 0.0
    th = 0.3

    
    device = "cuda:0"  if torch.cuda.is_available() else "cpu"
    model = YOLOv1(True)
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    xs, src_pad_and_resize_image = read_image_pad_and_resize_and_process("{}/{}.jpg".format(data_dir,image_number), image_size)
    xs_tensor = torch.tensor(xs, dtype=torch.float32).to(device)
    ys_tensor = model(xs_tensor)
    y0 = ys_tensor.cpu().detach().numpy()[0]


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
    # image = read_image("{}/{}.jpg".format(data_dir,image_number))
    image = src_pad_and_resize_image.copy()
    show_image_with_bboxs_and_names(image, one_data['objs'])
    cv2.imwrite("result/out.png", image)
    print("\n"*3)

    max_one_label = n.zeros((grid_number, grid_number, 25))
    if max_yx['y'] is not None:
        max_one_label[max_yx['y'], max_yx['x'], :] = max_label
    max_one_data = label_back_to_one_data(max_one_label, cls_th=cls_th)
    print(max_yx, max_label)
    print(max_one_data)
    print("max conf: ", max_label[0])
    # image = read_image("{}/{}.jpg".format(data_dir,image_number))
    image = src_pad_and_resize_image.copy()
    show_image_with_bboxs_and_names(image, max_one_data['objs'])
    print("\n"*3)


if __name__ == "__main__":
    test()









