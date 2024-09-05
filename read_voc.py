import xml.etree.ElementTree as ET
import cv2
import numpy as n
import os


voc_classes = [  
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',  
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',  
    'sheep', 'sofa', 'train', 'tvmonitor'  
]  
class_to_index = {cls: i for i, cls in enumerate(voc_classes)}  


def read_image(image_file):
    return cv2.imread(image_file)


def read_annotation(annotation_file):
    tree = ET.parse(annotation_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def read_annotation_with_class_and_position(annotation_file):

    one_data = {
        'size': {
            'height': None,
            'width': None
        },
        'objs': []
    }
    
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    size = root.find("size")
    one_data['size']['height'] = int(size.find("height").text)
    one_data['size']['width'] = int(size.find("width").text)

    for obj in root.findall('object'):
        dc = dict()
        dc['name'] = obj.find("name").text
        dc['class_id'] = class_to_index[obj.find("name").text]

        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        dc['box'] = [xmin, ymin, xmax, ymax]
        one_data['objs'].append(dc)
    return one_data


def class_balance(xml_dir):
    annotation_files = os.listdir(xml_dir)
    balance_weight = n.zeros((len(voc_classes)))
    balance_dict = {cls: 0 for i, cls in enumerate(voc_classes)}  
    for file in annotation_files:
        if file.endswith(".xml"):
            # print(file)
            anno = read_annotation_with_class_and_position(os.path.join(xml_dir, file))
            for obj in anno['objs']:
                balance_weight[obj['class_id']] += 1
                balance_dict[obj['name']] += 1
    balance_weight = 1 / (balance_weight + 1) 
    balance_weight = balance_weight * (1/n.max(balance_weight))
    # print(blance_weight)
    # print(blance_dict)
    return balance_weight


def class_balance_random(xml_dir):
    import random
    annotation_files = os.listdir(xml_dir)
    balance_weight = n.zeros((len(voc_classes)))
    balance_dict = {cls: 0 for i, cls in enumerate(voc_classes)}  
    for file in random.sample(annotation_files, 5000):
        if file.endswith(".xml"):
            # print(file)
            anno = read_annotation_with_class_and_position(os.path.join(xml_dir, file))
            for obj in anno['objs']:
                balance_weight[obj['class_id']] += 1
                balance_dict[obj['name']] += 1
    balance_weight = 1 / (balance_weight + 1) 
    balance_weight = balance_weight * (1/n.max(balance_weight))
    # print(blance_weight)
    # print(blance_dict)
    return balance_weight



if __name__ == "__main__":
    balance_weight = class_balance_random("./aug")
    print(balance_weight)
    balance_weight = class_balance("./aug")
    print(balance_weight)








