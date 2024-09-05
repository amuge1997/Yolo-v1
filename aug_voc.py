import os
import xml.etree.ElementTree as ET
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from read_voc import read_annotation, read_image


def write_pair(from_annotation_file, target_anotation_file, target_image_file, target_image, target_image_size, boxes):
    tree = ET.parse(from_annotation_file)
    root = tree.getroot()
    size = root.find('size')
    size.find("height").text = str(target_image_size[0])
    size.find("width").text = str(target_image_size[1])
    for obj, box in zip(root.findall('object'), boxes):
        x1, y1, x2, y2 = box
        obj.find('bndbox/xmin').text = str(int(x1))
        obj.find('bndbox/ymin').text = str(int(y1))
        obj.find('bndbox/xmax').text = str(int(x2))
        obj.find('bndbox/ymax').text = str(int(y2))
    tree.write(target_anotation_file)
    cv2.imwrite(target_image_file, target_image)

def augment_image(image, boxes, crop_size):
    seq_fix = iaa.Sequential([
        iaa.PadToFixedSize(width=crop_size, height=crop_size),
        iaa.CropToFixedSize(width=crop_size, height=crop_size)
    ])
    seq_random = iaa.Sequential([
        iaa.Affine(rotate=(-90, 90)),
        iaa.Fliplr(0.5),
        # 添加高斯噪声
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),
        # 添加高斯模糊
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.5))),
        # 随机色调变换
        iaa.Sometimes(0.5, iaa.AddToHueAndSaturation((-30, 30))),
        # 随机亮度变换
        iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.5, 1.5))),
        # 随机对比度变换
        iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 1.5))),
        # 随机缩放 (0.5x 到 1.5x)
        iaa.Sometimes(0.5, iaa.Affine(scale=(0.5, 1.5))),
        # 随机平移 (-20 到 20 像素)
        iaa.Sometimes(0.5, iaa.Affine(translate_px={"x": (-40, 40), "y": (-40, 40)})),
    ], random_order=True)

    seq = iaa.Sequential([
        seq_fix,
        seq_random,
    ])

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]) for box in boxes
    ], shape=image.shape)

    augmented_image, augmented_bbs = seq(image=image, bounding_boxes=bbs)
    
    # 更新边界框
    updated_boxes = []
    for bb in augmented_bbs:
        updated_boxes.append([bb.x1, bb.y1, bb.x2, bb.y2])

    return augmented_image, updated_boxes


def augment_pair_file(image_file, annotation_file, crop_size):
    image = read_image(image_file)
    boxes = read_annotation(annotation_file)
    augmented_image, updated_boxes = augment_image(image, boxes, crop_size)
    return augmented_image, updated_boxes


