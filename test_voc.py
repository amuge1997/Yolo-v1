import os
import xml.etree.ElementTree as ET
import cv2
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from aug_voc import augment_pair_file, write_pair
from global_param import image_size

dataset_dir = r'D:\soft\python_project\yolo\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'

annotations_dir = os.path.join(dataset_dir, 'Annotations')
images_dir = os.path.join(dataset_dir, 'JPEGImages')

annotation_files = os.listdir(annotations_dir)

annotation_files = ['005552.xml']
target_dir = "./aug"
# Process each annotation file
for annotation_file_src in annotation_files:
    image_file = os.path.join(images_dir, annotation_file_src[:-4] + '.jpg')
    annotation_file = os.path.join(annotations_dir, annotation_file_src)

    crop_size = image_size
    augmented_image, updated_boxes = augment_pair_file(image_file, annotation_file, crop_size=crop_size)

    augmented_image_save = augmented_image.copy()
    for box in updated_boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(augmented_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for i in range(13):
        cv2.line(augmented_image, (0, int(i*crop_size/13)), (crop_size, int(i*crop_size/13)), (255, 0, 0), 1)
        cv2.line(augmented_image, (int(i*crop_size/13), 0), (int(i*crop_size/13), crop_size), (255, 0, 0), 1)
    image_size = augmented_image.shape
    
    cv2.imshow('Augmented Image', augmented_image)
    cv2.waitKey(0)
    
    write_pair(
        from_annotation_file=annotation_file, 
        target_anotation_file=os.path.join(target_dir, annotation_file_src[:-4] + ".xml"),
        target_image_file=os.path.join(target_dir, annotation_file_src[:-4] + ".jpg"), 
        target_image=augmented_image_save, 
        target_image_size=image_size, 
        boxes=updated_boxes
    )

cv2.destroyAllWindows()






