import torch  

# box = [left_down.x, left_down.y, right_up.x, right_up.y]
def bbox_iou_xyxy(box1, box2):
    xi1 = torch.max(box1[0], box2[0])  
    yi1 = torch.max(box1[1], box2[1])  
    xi2 = torch.min(box1[2], box2[2])  
    yi2 = torch.min(box1[3], box2[3])  
    inter_width = torch.clamp(xi2 - xi1, min=0)  
    inter_height = torch.clamp(yi2 - yi1, min=0)  
    intersection = inter_width * inter_height  
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])  
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])  
    union = box1_area + box2_area - intersection   
    epsilon = 1e-7  
    iou = intersection / (union + epsilon)
    return iou

# box = [x_center, y_center, width, height]
def bbox_iou_xywh(box1, box2):
    box1 = [box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, box1[0] + box1[2] / 2, box1[1] + box1[3] / 2]
    box2 = [box2[0] - box2[2] / 2, box2[1] - box2[3] / 2, box2[0] + box2[2] / 2, box2[1] + box2[3] / 2]
    return bbox_iou_xyxy(box1, box2)

if __name__ == "__main__":
    box1 = torch.tensor([100.0, 100.0, 200.0, 200.0])  
    box2 = torch.tensor([150.0, 150.0, 250.0, 250.0])  
    iou_value = bbox_iou_xyxy(box1, box2)  
    print("IoU: {:.4f}".format(iou_value.item()))

    box1 = torch.tensor([150, 150, 100, 100])  
    box2 = torch.tensor([200, 200, 100, 100])  
    iou_value = bbox_iou_xywh(box1, box2)  
    print("IoU: {:.4f}".format(iou_value.item()))

    box1 = torch.tensor([15.36, 15.36, 48.64, 48.64])  
    box2 = torch.tensor([27.84, 27.84, 36.16, 36.16])  
    iou_value = bbox_iou_xyxy(box1, box2)  
    print("IoU: {:.4f}".format(iou_value.item()))









