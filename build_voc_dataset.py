from read_voc import read_annotation_with_class_and_position, read_image, voc_classes
from show_voc import show_image_with_bboxs
import numpy as n
from global_param import grid_size, image_size, grid_number


def label_back_to_one_data(labels, th=1e-5, cls_th=0.5):
    one_data = {
        'size': {
            'height': image_size,
            'width': image_size
        },
        'objs': []
    }
    for yi in range(grid_number):
        for xi in range(grid_number):
            label = labels[yi, xi]
            if label[0] < th:
                continue
            dc = dict()
            center_x = label[1] * grid_size + xi * grid_size
            center_y = label[2] * grid_size + yi * grid_size
            width = label[3] ** 2 * image_size
            height = label[4] ** 2 * image_size
            x0 = int(center_x - width / 2)
            y0 = int(center_y - height / 2)
            x1 = int(center_x + width / 2)
            y1 = int(center_y + height / 2)
            class_id = n.argmax(label[5:25])
            # print(yi, xi)
            # print(label)
            # print(label[5:25])
            # print()
            name = voc_classes[class_id]
            if label[class_id + 5] > cls_th:
                dc['name'] = name
                dc['class_id'] = class_id
                dc['box'] = [x0, y0, x1, y1]
                one_data['objs'].append(dc)
    return one_data

def build_label_by_one_data(one_data):
    # 如果一个grid中存在多个, 那么选择最后一个作为目标

    if one_data['size']['height'] != image_size:
        raise Exception('高度错误')
    if one_data['size']['width'] != image_size:
        raise Exception('宽度错误')
    
    one_data_label = n.zeros((grid_number, grid_number, 25))

    for obj in one_data['objs']:
        label = n.zeros((25,))
        conf = 1

        x0, y0, x1, y1 = obj['box']
        center_x = (x0 + x1) / 2
        grid_x = int(center_x // grid_size)
        center_x = center_x % grid_size / grid_size
        center_y = (y0 + y1) / 2
        grid_y = int(center_y // grid_size)
        center_y = center_y % grid_size / grid_size
        width = n.sqrt((x1 - x0) / image_size)
        height = n.sqrt((y1 - y0) / image_size)

        onehot = n.eye(20)[obj['class_id']:obj['class_id']+1]

        label[0] = conf
        label[1:5] = n.array([center_x, center_y, width, height])
        label[5:25] = onehot[0]

        if 0<=grid_y< grid_number and 0<=grid_x< grid_number:
            one_data_label[grid_y][grid_x] = label

    return one_data_label



if __name__ == "__main__":
    one_data = read_annotation_with_class_and_position("./aug_999999/112309.xml")
    print(one_data)

    # image = read_image("./aug/005552.jpg")
    # show_image_with_bboxs(image, [obj['box'] for obj in one_data['objs']])

    one_data_label = build_label_by_one_data(one_data)
    print(one_data_label[5][4])
    print(one_data_label.shape)

    back_one_data = label_back_to_one_data(one_data_label)
    print(back_one_data)
    
    image = read_image("./aug_999999/112309.jpg")
    show_image_with_bboxs(image, [obj['box'] for obj in back_one_data['objs']])














