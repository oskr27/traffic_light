from pathlib import Path
from PIL import Image
from random import choices

import random
import csv
import os


# Global variables
master_path = None
path_anno = None
path_img = None
path_output = None
day_clip_list = []
anno_file = None


# Global variable initialization
def initialize():
    global master_path
    master_path = Path('../../data/lisa-traffic-light-dataset')

    global path_anno
    path_anno = master_path / 'annotations'

    global path_img
    path_img = master_path / 'images'

    global path_output
    path_output = Path('../../data/cropped-dataset-no-filter')

    global day_clip_list
    day_clip_list = ['dayClip' + str(i) for i in range(1, 13)]

    global anno_file
    anno_file = Path('frameAnnotationsBOX.csv')


# Definition of area normalization function
# The box is a tuple of four integers, x1, y1, x2, and y2
# To obtain a normal area for this box, it is necessary to get the
# center of the area, and expand the area to a 224x224 square shape
def get_normal_area(box):
    # Getting center of image
    x_center = (box[0] + box[2]) // 2
    y_center = (box[1] + box[3]) // 2

    # Creating new tuple with expanded area
    x_1 = x_center - 111;
    y_1 = y_center - 111;

    x_2 = x_center + 112 if x_1 >= 0 else x_center + 112 - x_1
    y_2 = y_center + 112 if y_1 >= 0 else y_center + 112 - y_1

    x_1 = x_1 if x_1 >= 0 else 0
    y_1 = y_1 if y_1 >= 0 else 0

    return x_1, y_1, x_2, y_2


# Definition of random path generator. This will return the output
# file path (train or valid). The input parameters will
# keep track of the distribution of these.
def get_random_path():
    opts = ['train', 'valid', 'test']
    prob = [0.7, 0.2, 0.1]

    return choices(opts, prob)[0]


# Returns the image class depending on the input
# The "flag" parameter is to condense the warning signal with stop
def get_image_class(image_class, flag):
    if flag:
        if image_class == 'warning':
            return 'stop'
        elif image_class == 'warningLeft':
            return 'stopLeft'
        else:
            return image_class
    else:
        return image_class


def image_cropper():
    for dayClip in day_clip_list:
        clip_annotations = path_anno / 'dayTrain' / dayClip / anno_file
        clip_images = path_img / 'dayTrain' / dayClip / 'frames'

        with open(clip_annotations) as csv_file:
            reader = csv.reader(csv_file, delimiter=';')
            # The following line skips the headers
            next(reader)

            image_number = []
            file_name = []
            tag = []
            box = []

            i = 0
            for row in reader:
                image_number.append(i)
                file_name.append(row[0].replace('dayTraining/', ''))
                tag.append(row[1])
                box.append(row[2:6])
                i = i + 1

        i = 0
        while i < len(image_number):
            image = Image.open(clip_images / file_name[i])

            area = get_normal_area(tuple(map(int, box[i])))
            cropped_image = image.crop(area)

            random_path = Path(get_random_path())
            path_image_output = path_output / random_path / get_image_class(tag[i], True)

            if not os.path.exists(path_image_output):
                os.makedirs(path_image_output)

            cropped_image.save(path_image_output / Path(dayClip + '_' + str(image_number[i]) + '.jpg'))

            i = i + 1

def main():
    initialize()
    image_cropper()


if __name__ == '__main__':
    main()
