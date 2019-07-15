import cv2
import os
import yaml
import random
import argparse

WIDTH = 1280
HEIGHT = 720


def is_eligible(box):
    if box['occluded'] == 'True':
        return False
    elif ir(box['y_max']) - ir(box['y_min']) < 35:
        return False
    elif 'Right' in box['label']:
        return False
    elif 'Straight' in box['label']:
        return False
    elif box['label'] == 'off':
        return False
    else:
        return True


def get_normal_label(box):
    label = box['label']

    if 'Green' in label:
        return str.replace(label, 'Green', 'go')
    elif 'Yellow' in label:
        return str.replace(label, 'Yellow', 'warning')
    elif 'Red' in label:
        return str.replace(label, 'Red', 'stop')
    else:
        return label


def get_random_path():
    opts = ['train', 'valid', 'test']
    prob = [0.7, 0.2, 0.1]

    return random.choices(opts, prob)[0]


def get_normal_area(box):
    # Getting center of image
    x_center = (box[0] + box[2]) // 2
    y_center = (box[1] + box[3]) // 2

    # Creating new tuple with expanded area
    x_1 = x_center - 111
    y_1 = y_center - 111

    x_2 = x_center + 112 if x_1 >= 0 else x_center + 112 - x_1
    y_2 = y_center + 112 if y_1 >= 0 else y_center + 112 - y_1

    x_1 = x_1 if x_1 >= 0 else 0
    y_1 = y_1 if y_1 >= 0 else 0

    x_2 = x_2 if x_2 < WIDTH else WIDTH - 1
    y_2 = y_2 if y_2 < HEIGHT else HEIGHT - 1

    return x_1, y_1, x_2, y_2


def ir(some_value):
    return int(round(some_value))


# This code comes from the official repository of the BSTLD dataset. Please refer
# to its own documentation to have better understanding of all the features available.
def get_all_labels(input_yaml, clip=True):
    assert os.path.isfile(input_yaml), 'Input yaml {} does not exist'.format(input_yaml)
    with open(input_yaml, 'rb') as iy_handle:
        images = yaml.load(iy_handle, Loader=yaml.FullLoader)

    if not images or not isinstance(images[0], dict) or 'path' not in images[0]:
        raise ValueError('Something seems wrong with this label-file: {}'.format(input_yaml))

    for i in range(len(images)):
        images[i]['path'] = os.path.abspath(os.path.join(os.path.dirname(input_yaml),
                                                         images[i]['path']))

        for j, box in enumerate(images[i]['boxes']):
            if box['x_min'] > box['x_max']:
                images[i]['boxes'][j]['x_min'], images[i]['boxes'][j]['x_max'] = (
                    images[i]['boxes'][j]['x_max'], images[i]['boxes'][j]['x_min'])

            if box['y_min'] > box['y_max']:
                images[i]['boxes'][j]['y_min'], images[i]['boxes'][j]['y_max'] = (
                    images[i]['boxes'][j]['y_max'], images[i]['boxes'][j]['y_min'])

        if clip:
            for j, box in enumerate(images[i]['boxes']):
                images[i]['boxes'][j]['x_min'] = max(min(box['x_min'], WIDTH - 1), 0)
                images[i]['boxes'][j]['x_max'] = max(min(box['x_max'], WIDTH - 1), 0)
                images[i]['boxes'][j]['y_min'] = max(min(box['y_min'], HEIGHT - 1), 0)
                images[i]['boxes'][j]['y_max'] = max(min(box['y_max'], HEIGHT - 1), 0)
    return images


def crop_images_from_labels(input_yaml, output_folder):
    images = get_all_labels(input_yaml)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, image_dict in enumerate(images):
        image = cv2.imread(image_dict['path'])

        if image is None:
            raise IOError('Could not open image path', image_dict['path'])

        j = 0
        for box in image_dict['boxes']:
            write = is_eligible(box)

            my_box = get_normal_area((ir(box['x_min']), ir(box['y_min']), ir(box['x_max']), ir(box['y_max'])))
            crop_img = image[my_box[1]:my_box[3], my_box[0]:my_box[2]].copy()

            if write:
                random_path = get_random_path()

                if random_path != 'test':
                    new_path = os.path.join(output_folder, random_path, get_normal_label(box))
                else:
                    new_path = os.path.join(output_folder, random_path)

                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                cv2.imwrite(os.path.join(new_path
                                         , str(i).zfill(5) + '_'
                                         + str(j) + '_' + os.path.basename(image_dict['path'])), crop_img)
            j = j + 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default='')
    parser.add_argument('--output_dir', default='../../data/bosch-cropped-dataset')
    return parser.parse_args()


def main(args):
    crop_images_from_labels(args.yaml_file, args.output_dir)


if __name__ == '__main__':
    main(parse_args())
