import numpy as n
import torch
from global_param import grid_size, image_size
import torch.nn as nn
import torch.nn.functional as F


def cal_iou(predict_xywh, label_xywh):
    device = "cuda:0"  if torch.cuda.is_available() else "cpu"
    # predict_xywh.shape = batch, 13, 13, 2, 4
    # label_xywh.shape   = batch, 13, 13, 2, 4
    predict_xyxy = torch.zeros_like(predict_xywh, dtype=torch.float32, device=device)
    predict_xyxy[..., 0] = predict_xywh[..., 0] * grid_size - torch.pow(predict_xywh[..., 2], 2) * image_size
    predict_xyxy[..., 1] = predict_xywh[..., 1] * grid_size - torch.pow(predict_xywh[..., 3], 2) * image_size
    predict_xyxy[..., 2] = predict_xywh[..., 0] * grid_size + torch.pow(predict_xywh[..., 2], 2) * image_size
    predict_xyxy[..., 3] = predict_xywh[..., 1] * grid_size + torch.pow(predict_xywh[..., 3], 2) * image_size
    label_xyxy = torch.zeros_like(label_xywh, dtype=torch.float32, device=device)
    label_xyxy[..., 0] = label_xywh[..., 0] * grid_size - torch.pow(label_xywh[..., 2], 2) * image_size
    label_xyxy[..., 1] = label_xywh[..., 1] * grid_size - torch.pow(label_xywh[..., 3], 2) * image_size
    label_xyxy[..., 2] = label_xywh[..., 0] * grid_size + torch.pow(label_xywh[..., 2], 2) * image_size
    label_xyxy[..., 3] = label_xywh[..., 1] * grid_size + torch.pow(label_xywh[..., 3], 2) * image_size
    
    mx = torch.maximum(predict_xyxy, label_xyxy)    # batch, 13, 13, 2, 4
    mn = torch.minimum(predict_xyxy, label_xyxy)    # batch, 13, 13, 2, 4
    x0 = mx[..., 0:1]               # batch, 13, 13, 2, 1
    y0 = mx[..., 1:2]               # batch, 13, 13, 2, 1
    x1 = mn[..., 2:3]               # batch, 13, 13, 2, 1
    y1 = mn[..., 3:4]               # batch, 13, 13, 2, 1
    xd = x1 - x0                    # batch, 13, 13, 2, 1
    yd = y1 - y0                    # batch, 13, 13, 2, 1
    xd = torch.where(xd < 0., torch.tensor(0., device=device), xd)    # batch, 13, 13, 2, 1
    yd = torch.where(yd < 0., torch.tensor(0., device=device), yd)    # batch, 13, 13, 2, 1

    intersection = xd * yd          # batch, 13, 13, 2, 1

    predict_area = (predict_xyxy[..., 2:3] - predict_xyxy[..., 0:1]) * (predict_xyxy[..., 3:4] - predict_xyxy[..., 1:2])            # batch, 13, 13, 2, 1
    label_rep_area = (label_xyxy[..., 2:3] - label_xyxy[..., 0:1]) * (label_xyxy[..., 3:4] - label_xyxy[..., 1:2])                  # batch, 13, 13, 2, 1

    ret = intersection / (predict_area + label_rep_area - intersection + 1e-5)             # batch, 13, 13, 2, 1
    return ret


bce = nn.BCELoss()

def yolo_loss_sigmoid(predict, label, balance_weight):
    obj_scale = 5 * 1
    noobj_scale = 0.5
    
    conf_scale = 1.
    noconf_scale = 1.
    xy_scale = 1.
    wh_scale = 2.
    class_scale = 1.

    predict_conf = predict[:, :, :, 0:2]        # batch, 13, 13, 2
    label_conf = label[:, :, :, 0:1]            # batch, 13, 13, 1

    batch, grid_y, gird_x = predict.shape[0:3]
    predict_xywh = predict[:, :, :, 2:10].reshape(batch, grid_y, gird_x, 2, 4)  # batch, 13, 13, 2, 4
    predict_class =  predict[:, :, :, 10:]      # batch, 13, 13, 20
    
    label_xywh = label[:, :, :, 1:5].reshape(batch, grid_y, gird_x, 1, 4)       # batch, 13, 13, 1, 4
    label_xywh = label_xywh.repeat(1, 1, 1, predict_xywh.shape[-2], 1)          # batch, 13, 13, 2, 4

    iou = cal_iou(predict_xywh, label_xywh)     # batch, 13, 13, 2, 1
    
    label_class = label[:, :, :, 5:]          # batch, 13, 13, 20

    obj_mask_max, _ = torch.max(iou, dim=3, keepdim=True)       # batch, 13, 13, 1, 1
    obj_mask = iou
    obj_mask = obj_mask >= obj_mask_max.repeat(1, 1, 1, predict_xywh.shape[-2], 1)                  # batch, 13, 13, 2, 1

    obj_mask = obj_mask * label_conf.reshape(batch, grid_y, gird_x, 1, 1).repeat(1, 1, 1, predict_xywh.shape[-2], 1)                # batch, 13, 13, 2, 1

    noobj_mask = 1 - obj_mask                   # batch, 13, 13, 2, 1

    obj_conf_loss = bce(predict_conf[..., n.newaxis], obj_mask) * obj_mask            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    obj_conf_loss = obj_conf_loss.sum() / torch.sum(obj_mask == 1)
    obj_conf_loss = obj_conf_loss * obj_scale * conf_scale
    
    noobj_conf_loss = bce(predict_conf[..., n.newaxis], torch.zeros_like(predict_conf[..., n.newaxis])) * noobj_mask
    noobj_conf_loss = noobj_conf_loss.sum() / torch.sum(noobj_mask == 1)
    noobj_conf_loss = noobj_conf_loss * noobj_scale * noconf_scale
    
    obj_xy_loss = bce(predict_xywh[:,:,:,:,0:2], label_xywh[:,:,:,:,0:2]) * obj_mask
    obj_xy_loss = obj_xy_loss.sum() / torch.sum(obj_mask == 1)
    obj_xy_loss = obj_xy_loss * obj_scale * xy_scale

    obj_wh_loss = obj_mask * (label_xywh[:,:,:,:,2:4] - predict_xywh[:,:,:,:,2:4])
    obj_wh_loss = (obj_wh_loss ** 2)
    obj_wh_loss = obj_wh_loss.sum() / torch.sum(obj_mask == 1)
    obj_wh_loss = obj_wh_loss * obj_scale * wh_scale
    
    b = torch.where(label_class == 1)
    balance_weight_exp = balance_weight.repeat(label_class.shape[0], label_class.shape[1], label_class.shape[2], 1)
    balance_weight_ones = torch.ones_like(balance_weight_exp)
    balance_weight_ones[b[0], b[1], b[2], b[3]] = balance_weight_exp[b[0], b[1], b[2], b[3]] * label_class.shape[3]
    obj_class_loss = bce(predict_class, label_class) * balance_weight_ones           # batch, 13, 13, 20
    obj_class_loss = obj_class_loss[(label_conf == 1).repeat(1, 1, 1, label_class.shape[3])]
    obj_class_loss = obj_class_loss.mean()
    obj_class_loss = obj_class_loss * obj_scale * class_scale
    
    loss = obj_conf_loss + obj_xy_loss + obj_wh_loss + obj_class_loss + noobj_conf_loss
    return loss, obj_conf_loss, obj_xy_loss,obj_wh_loss, obj_class_loss, noobj_conf_loss


def test_iou():
    
    A = torch.zeros((10, 13, 13, 30))
    A[0, 0, 0, 0] = 1.
    A[0, 0, 0, 2] = 1.
    A[0, 0, 0, 3] = 1.
    A[0, 0, 0, 4] = 0.2
    A[0, 0, 0, 5] = 0.2
    A[0, 0, 0, 1] = 1.
    A[0, 0, 0, 5+5] = 0.6

    A[0, 1, 0, 0] = 1.
    A[0, 1, 0, 4] = 0.1
    A[0, 1, 0, 5] = 0.1
    B = torch.zeros((10, 13, 13, 25))
    B[0, 0, 0, 0] = 1.
    B[0, 0, 0, 1] = 1.
    B[0, 0, 0, 2] = 1.
    B[0, 0, 0, 3] = 0.1
    B[0, 0, 0, 4] = 0.1

    B[0, 1, 0, 0] = 1.
    B[0, 1, 0, 3] = 0.1
    B[0, 1, 0, 4] = 0.1
    B[0, 1, 0, 5] = 1.
    B[0, 1, 0, 5+7] = 1.
    B[0, 1, 0, 5+11] = 1.

    from read_voc import class_balance
    
    balance_weight = class_balance("./aug")
    balance_weight = torch.tensor(balance_weight, dtype=torch.float32)
    loss = yolo_loss_sigmoid(A, B, balance_weight)
    # print(loss)


if __name__ == "__main__":
    test_iou()
    










