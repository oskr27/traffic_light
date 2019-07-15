import cv2
import yaml
import numpy as np
import os.path
import copy
import argparse

WIDTH = 2048
HEIGHT = 1024


def ir(some_value):
    return int(round(some_value))


def get_random_path():
    opts = ['train', 'valid', 'test']
    prob = [0.7, 0.2, 0.1]

    return np.random.choice(opts, size=1, p=prob)[0]


class DriveuObject:
    """ Class describing a label object in the dataset by rectangle

    Attributes:
        x (int):          X coordinate of upper left corner of bounding box label
        y (int):          Y coordinate of upper left corner of bounding box label
        width (int):      Width of bounding box label
        height (int):     Height of bounding box label
        class_id (int):   6 Digit class identity of bounding box label (Digit explanation see documentation pdf)
        unique_id (int):  Unique ID of the object
        track_id (string) Track ID of the object (representing one real-world TL instance)

    """
    def __init__(self):
        self.x = 0
        self.y = 0
        self.width = 0
        self.height = 0
        self.class_id = 0
        self.unique_id = 0
        self.track_id = 0

    def color_from_class_id(self):
        """ Color for bounding box visualization

            Returns:
            Color-Vector (BGR) for traffic light visualization

        """
        # Second last digit indicates state/color
        if str(self.class_id)[-2] == "1":
            return 0, 0, 255
        elif str(self.class_id)[-2] == "2":
            return 0, 255, 255
        elif str(self.class_id)[-2] == "3":
            return 0, 165, 255
        elif str(self.class_id)[-2] == "4":
            return 0, 255, 0
        else:
            return 255, 255, 255

    def get_normal_area(self):
        # Getting center of image
        x_center = (self.x + self.x + self.width) // 2
        y_center = (self.y + self.y + self.height) // 2

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

    def get_normal_label(self):
        if str(self.class_id)[-2] == "1":
            label = 'stop'
        elif str(self.class_id)[-2] == "2":
            label = 'warning'
        elif str(self.class_id)[-2] == "3":
            label = 'stop'
        elif str(self.class_id)[-2] == "4":
            label = 'go'
        else:
            label = 'unknown'

        if str(self.class_id)[-1] == "0" or str(self.class_id)[-1] == "1":
            label = label
        elif str(self.class_id)[-1] == "2":
            label = label + 'Left'
        elif str(self.class_id)[-1] == "3":
            label = label + 'Left'
        elif str(self.class_id)[-1] == "4":
            label = label + 'Right'
        else:
            label = label + 'Others'

        return label

    def is_eligible(self):
        if str(self.class_id)[0] != '1':
            return False
        if str(self.class_id)[1] == '2' or str(self.class_id)[1] == '3':
            return False
        elif str(self.class_id)[-2] == '0':
            return False
        elif str(self.class_id)[-1] == '8' or str(self.class_id)[-1] == '9':
            return False
        elif self.height < 35:
            return False
        else:
            return True


class DriveuImage:
    """ Class describing one image in the DriveU Database

    Attributes:
        file_path (string):         Path of the left camera image
        timestamp (float):          Timestamp of the image
        objects (DriveuObject)      Labels in that image
    """
    def __init__(self):
        self.file_path = ''
        self.timestamp = 0
        self.objects = []

    def get_image(self):
        """ Method loading the left unrectified color image in 8 Bit

        Returns:
            8 Bit BGR color image

        """
        if os.path.isfile(self.file_path):
            """Load image from file path, do debayering and shift"""
            img = cv2.imread(self.file_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)

            # Images are saved in 12 bit raw -> shift 4 bits
            img = np.right_shift(img, 4)
            img = img.astype(np.uint8)

            return True, img
        else:
            print("Image " + str(self.file_path) + "not found")
            return False, img

    def get_labeled_image_and_crop(self, output_folder):
        """Method loading the left unrectified color image and drawing labels in it

        Returns:
            Labeled 8 Bit BGR color image

        """
        status, img = self.get_image()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for o in self.objects:
            # Save image in output_folder
            write = o.is_eligible()

            my_box = o.get_normal_area()
            crop_img = img[my_box[1]:my_box[3], my_box[0]:my_box[2]].copy()

            if write:
                random_path = get_random_path()

                if random_path != 'test':
                    new_path = os.path.join(output_folder, random_path, o.get_normal_label())
                else:
                    new_path = os.path.join(output_folder, random_path)

                if not os.path.exists(new_path):
                    os.makedirs(new_path)

                cv2.imwrite(os.path.join(new_path, str(o.unique_id) + '_' + str(o.class_id) + '.png'), crop_img)


class DriveuDatabase:
    """ Class describing the DriveU Dataset

    Attributes:
        images (List of DriveuImage)  All images of the dataset
        file_path (string):           Path of the dataset (.yml)
    """
    def __init__(self, file_path):
        self.images = []
        self.file_path = file_path

    def open(self, data_base_dir):
        """Method loading the dataset"""
        if os.path.exists(self.file_path) is not None:
            print('Opening DriveuDatabase from File: ' + str(self.file_path))
            images = yaml.load(open(self.file_path, 'rb').read(), Loader=yaml.FullLoader)

            for i, image_dict in enumerate(images):
                image = DriveuImage()
                if data_base_dir != '':
                    inds = [i for i, c in enumerate(image_dict['path']) if c == '/']
                    image.file_path = data_base_dir + image_dict['path'][inds[-4]:]
                    inds = [i for i, c in enumerate(image_dict['disp_path']) if c == '/']
                    image.disp_file_path = data_base_dir + '/' + image_dict['disp_path'][inds[-4]:]
                else:
                    image.file_path = image_dict['path']
                    image.disp_file_path = image_dict['disp_path']
                image.timestamp = image_dict['time_stamp']

                for o in image_dict['objects']:
                    new_object = DriveuObject()

                    new_object.x = o['x']
                    new_object.y = o['y']
                    new_object.width = o['width']
                    new_object.height = o['height']
                    new_object.class_id = o['class_id']
                    new_object.unique_id = o['unique_id']
                    new_object.track_id = o['track_id']

                    cpy = copy.copy(new_object)

                    image.objects.append(cpy)

                copy_image = copy.copy(image)
                self.images.append(copy_image)
        else:
            print('Opening DriveuDatabase from File: ' + str(self.file_path) + 'failed. File or Path incorrect.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_file', default='')
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--output_dir', default='../../data/dtld-cropped-dataset')
    return parser.parse_args()


def main(args):
    database = DriveuDatabase(args.yaml_file)
    database.open(args.dataset_dir)

    for idx_d, img in enumerate(database.images):
        img.get_labeled_image_and_crop(args.output_dir)


if __name__ == '__main__':
    main(parse_args())
