import xml.etree.ElementTree as ET
import cv2
from read_voc import read_annotation
from global_param import image_size, grid_number, grid_size
crop_size = image_size

def show_image_with_bboxs(image, boxes):
    # bbox = [[x0,y0,x1,y1]]
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)

def show_image_with_bboxs_and_names(image, one_data):
    # bbox = [[x0,y0,x1,y1]]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for obj in one_data:
        box = obj['box']
        name = obj['name']
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, name, (int(x1), int(y1)), font, 1, (0, 255, 0), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)

def show_pascal_voc(image_file, annotation_file, is_show_box, is_show_grid=False):
    image = cv2.imread(image_file)
    if (is_show_box):
        boxes = read_annotation(annotation_file)
        for box in boxes:
            x1, y1, x2, y2 = box
            print((x1+x2)/2 // grid_size, (y1+y2)/2 // grid_size)
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for i in range(grid_number):
        cv2.line(image, (0, int(i*crop_size/grid_number)), (crop_size, int(i*crop_size/grid_number)), (255, 0, 0), 1)
        cv2.line(image, (int(i*crop_size/grid_number), 0), (int(i*crop_size/grid_number), crop_size), (255, 0, 0), 1)
    cv2.imshow('Pascal Voc Image', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    # show_pascal_vol(r"./VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages\000009.jpg", r"VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007\Annotations\000009.xml")
    # show_pascal_voc("aug/005552.jpg", "aug/005552.xml", True)
    show_pascal_voc("aug_only_person/705171.jpg", "aug_only_person/705171.xml", True, True)








