import os
import xml.etree.ElementTree as ET
import cv2
import numpy as n
from aug_voc import augment_pair_file, write_pair
from build_voc_dataset import read_annotation_with_class_and_position


def make_aug_voc_dataset():
    # dataset_dir = r'..\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    dataset_dir = r'..\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'

    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    images_dir = os.path.join(dataset_dir, 'JPEGImages')

    annotation_files = os.listdir(annotations_dir)
    
    # annotation_files = ['005552.xml']
    target_dir = "./aug"
    make_data_number = int((50000 * 2 - 99998) / 2)

    # Process each annotation file
    make_number = 0
    while True:
        n.random.shuffle(annotation_files)
        for j, annotation_file_src in enumerate(annotation_files):
            if make_number >= make_data_number:
                return
            make_number += 1
            try:
                image_file = os.path.join(images_dir, annotation_file_src[:-4] + '.jpg')
                annotation_file = os.path.join(annotations_dir, annotation_file_src)

                from global_param import image_size as crop_size
                from global_param import grid_number
                augmented_image, updated_boxes = augment_pair_file(image_file, annotation_file, crop_size=crop_size)
                augmented_image_save = augmented_image.copy()
                for box in updated_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(augmented_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                for i in range(grid_number):
                    cv2.line(augmented_image, (0, int(i*crop_size/grid_number)), (crop_size, int(i*crop_size/grid_number)), (255, 0, 0), 1)
                    cv2.line(augmented_image, (int(i*crop_size/grid_number), 0), (int(i*crop_size/grid_number), crop_size), (255, 0, 0), 1)
                image_size = augmented_image.shape
                
                
                name = n.random.randint(100000, 999999)
                write_pair(
                    from_annotation_file=annotation_file, 
                    target_anotation_file=os.path.join(target_dir, str(name) + ".xml"),
                    target_image_file=os.path.join(target_dir, str(name) + ".jpg"), 
                    target_image=augmented_image_save, 
                    target_image_size=image_size, 
                    boxes=updated_boxes
                )
                print("{}/{}".format(make_data_number, make_number))
            except:
                pass
            #     cv2.imshow('Augmented Image', augmented_image)
            #     cv2.waitKey(0)

            # cv2.destroyAllWindows()



if __name__ == "__main__":
    make_aug_voc_dataset()


