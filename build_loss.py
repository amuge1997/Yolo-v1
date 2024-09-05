import numpy as n
from iou_torch import bbox_iou_xywh

image_size = 416
grid_number = 13
grid_size = image_size / grid_number

def cal_iou(predict_xywh, label_xywh):
    # predict_xywh.shape = batch, 13, 13, 2, 4
    # label_xywh.shape   = batch, 13, 13, 2, 4
    predict_xyxy = n.zeros_like(predict_xywh)
    predict_xyxy[..., 0] = predict_xywh[..., 0] * grid_size - n.power(predict_xywh[..., 2], 2) * image_size
    predict_xyxy[..., 1] = predict_xywh[..., 1] * grid_size - n.power(predict_xywh[..., 3], 2) * image_size
    predict_xyxy[..., 2] = predict_xywh[..., 0] * grid_size + n.power(predict_xywh[..., 2], 2) * image_size
    predict_xyxy[..., 3] = predict_xywh[..., 1] * grid_size + n.power(predict_xywh[..., 3], 2) * image_size
    label_xyxy = n.zeros_like(label_xywh)
    label_xyxy[..., 0] = label_xywh[..., 0] * grid_size - n.power(label_xywh[..., 2], 2) * image_size
    label_xyxy[..., 1] = label_xywh[..., 1] * grid_size - n.power(label_xywh[..., 3], 2) * image_size
    label_xyxy[..., 2] = label_xywh[..., 0] * grid_size + n.power(label_xywh[..., 2], 2) * image_size
    label_xyxy[..., 3] = label_xywh[..., 1] * grid_size + n.power(label_xywh[..., 3], 2) * image_size
    
    print('iou0000')
    print(predict_xywh[0,0,0,0])
    print(predict_xyxy[0,0,0,0])
    print(label_xywh[0,0,0,0])
    print(label_xyxy[0,0,0,0])
    print()

    # label_xyxy_rep = n.repeat(label_xywh, predict_xyxy.shape[-2], 3)                # batch, 13, 13, 2, 4
    mx = n.maximum(predict_xyxy, label_xyxy)    # batch, 13, 13, 2, 4
    mn = n.minimum(predict_xyxy, label_xyxy)    # batch, 13, 13, 2, 4
    x0 = mx[..., 0:1]               # batch, 13, 13, 2, 1
    y0 = mx[..., 1:2]               # batch, 13, 13, 2, 1
    x1 = mn[..., 2:3]               # batch, 13, 13, 2, 1
    y1 = mn[..., 3:4]               # batch, 13, 13, 2, 1
    xd = x1 - x0                    # batch, 13, 13, 2, 1
    yd = y1 - y0                    # batch, 13, 13, 2, 1

    xd = n.where(xd < 0, 0., xd)    # batch, 13, 13, 2, 1
    yd = n.where(yd < 0, 0., yd)    # batch, 13, 13, 2, 1

    intersection = xd * yd          # batch, 13, 13, 2, 1

    predict_area = (predict_xyxy[..., 2:3] - predict_xyxy[..., 0:1]) * (predict_xyxy[..., 3:4] - predict_xyxy[..., 1:2])            # batch, 13, 13, 2, 1
    label_rep_area = (label_xyxy[..., 2:3] - label_xyxy[..., 0:1]) * (label_xyxy[..., 3:4] - label_xyxy[..., 1:2])                  # batch, 13, 13, 2, 1

    ret = intersection / (predict_area + label_rep_area - intersection + 1e-5)             # batch, 13, 13, 2, 1
    return ret


def MSE():
    pass


def yolo_loss(predict, label):
    # predict.shape = batch, 13, 13, 30
    # label.shape   = batch, 13, 13, 25

    obj_scale = 5
    noobj_scale = 0.5

    predict_conf = predict[:, :, :, 0:2]        # batch, 13, 13, 2
    label_conf = label[:, :, :, 0:1]            # batch, 13, 13, 1

    batch, grid_y, gird_x = predict.shape[0:3]
    predict_xywh = predict[:, :, :, 2:10].reshape(batch, grid_y, gird_x, 2, 4)  # batch, 13, 13, 2, 4
    predict_class =  predict[:, :, :, 10:]      # batch, 13, 13, 20
    
    label_xywh = label[:, :, :, 1:5].reshape(batch, grid_y, gird_x, 1, 4)       # batch, 13, 13, 1, 4
    label_xywh = n.repeat(label_xywh, repeats=predict_xywh.shape[-2], axis=3)   # batch, 13, 13, 2, 4
    iou = cal_iou(predict_xywh, label_xywh)     # batch, 13, 13, 2, 1
    print('iou')
    print(iou[0,0,0,0])
    print()
    
    label_class = label[:, :, :, 5:25]          # batch, 13, 13, 20

    obj_mask_max = n.max(iou, axis=3, keepdims=True)
    obj_mask = iou
    obj_mask = obj_mask >= n.repeat(obj_mask_max, repeats=predict_xywh.shape[-2], axis=3)       # batch, 13, 13, 2, 1
    obj_mask = obj_mask * n.repeat(label_conf.reshape(batch, grid_y, gird_x, 1, 1), repeats=predict_xywh.shape[-2], axis=3)          # batch, 13, 13, 2, 1

    noobj_mask = 1 - obj_mask                   # batch, 13, 13, 2, 1
    # print(iou.shape, predict_conf.shape)
    # exit()
    obj_conf_loss = obj_mask * (iou - predict_conf[..., n.newaxis])             # batch, 13, 13, 2, 1
    print('obj_conf_loss')
    print(obj_mask[0,0,0,0], obj_conf_loss[0,0,0,0])
    print()
    obj_conf_loss = n.sqrt(obj_conf_loss * obj_conf_loss)
    obj_conf_loss = obj_conf_loss.mean()
    obj_conf_loss = obj_conf_loss * obj_scale

    noobj_conf_loss = noobj_mask * (0 - predict_conf[..., n.newaxis])           # batch, 13, 13, 2, 1
    # print(obj_mask.shape, predict_conf[..., n.newaxis].shape)
    print('noobj_conf_loss')
    print(noobj_mask[0,0,0,1], noobj_conf_loss[0,0,0,1])
    print()
    noobj_conf_loss = n.sqrt(noobj_conf_loss * noobj_conf_loss)
    noobj_conf_loss = noobj_conf_loss.mean()
    noobj_conf_loss = noobj_conf_loss * noobj_scale

    obj_xywh_loss = obj_mask * (label_xywh - predict_xywh)
    # print(obj_mask.shape, label_xywh.shape)
    print('obj_xywh_loss')
    print(obj_mask[0,0,0,0], obj_xywh_loss[0,0,0,0])
    print()
    obj_xywh_loss = n.sqrt(obj_xywh_loss * obj_xywh_loss)
    obj_xywh_loss = obj_xywh_loss.mean()
    obj_xywh_loss = obj_xywh_loss * obj_scale

    obj_class_loss = label_conf * (label_class - predict_class)             # batch, 13, 13, 20
    # print(label_conf.shape, (label_class - predict_class).shape)
    print('obj_xywh_loss')
    print(label_conf[0,0,0,0], obj_class_loss[0,0,0,0])
    print()
    obj_class_loss = n.sqrt(obj_class_loss * obj_class_loss)
    obj_class_loss = obj_class_loss.mean()
    obj_class_loss = obj_class_loss * obj_scale
    
    loss = obj_conf_loss + obj_xywh_loss + obj_class_loss + noobj_conf_loss
    return loss, iou[0,0,0,0]


def test_iou():
    A = n.zeros((10, 13, 13, 30))
    A[0, 0, 0, 0] = 1.
    A[0, 0, 0, 2] = 1.
    A[0, 0, 0, 3] = 1.
    A[0, 0, 0, 4] = 0.2
    A[0, 0, 0, 5] = 0.2
    A[0, 0, 0, 1] = 1.
    A[0, 0, 0, 5+5] = 0.6
    B = n.zeros((10, 13, 13, 25))
    B[0, 0, 0, 0] = 1.
    B[0, 0, 0, 1] = 1.
    B[0, 0, 0, 2] = 1.
    B[0, 0, 0, 3] = 0.1
    B[0, 0, 0, 4] = 0.1
    B[0, 0, 0, 5] = 1.
    
    loss, iou0000 = yolo_loss(A, B)
    print("{}?={}".format(0.0625, iou0000))


if __name__ == "__main__":
    test_iou()
    










