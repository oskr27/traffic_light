from pathlib import Path
import random
import argparse
import os
import shutil


def ir(some_value):
    return int(round(some_value))


def is_selected(probability):
    return random.random() < probability


def copy_images(lisa_dataset_dir, bosch_dataset_dir, dtld_dataset_dir, output_dir):
    images = []

    if lisa_dataset_dir != '':
        images += [file for file in lisa_dataset_dir.glob('**/*.*')]
    if bosch_dataset_dir != '':
        images += [file for file in bosch_dataset_dir.glob('**/*.*')]
    if dtld_dataset_dir != '':
        images += [file for file in dtld_dataset_dir.glob('**/*.*')]

    images = random.sample(images, ir(len(images) * 0.10))

    for image in images:
        if str(lisa_dataset_dir) in str(image):
            output = str.replace(str(image), str(lisa_dataset_dir), str(output_dir))
        elif str(bosch_dataset_dir) in str(image):
            output = str.replace(str(image), str(bosch_dataset_dir), str(output_dir))
        elif str(dtld_dataset_dir) in str(image):
            output = str.replace(str(image), str(dtld_dataset_dir), str(output_dir))

        if not os.path.exists(Path(output).parent):
            os.makedirs(Path(output).parent)

        shutil.copy(image, output)

        if is_selected(0.1):
            if str(lisa_dataset_dir) in str(image):
                sample_output = str.replace(str(image), str(lisa_dataset_dir), str(output_dir) + '-sample')
            elif str(bosch_dataset_dir) in str(image):
                sample_output = str.replace(str(image), str(bosch_dataset_dir), str(output_dir) + '-sample')
            elif str(dtld_dataset_dir) in str(image):
                sample_output = str.replace(str(image), str(dtld_dataset_dir), str(output_dir) + '-sample')

            if not os.path.exists(Path(sample_output).parent):
                os.makedirs(Path(sample_output).parent)

            shutil.copy(image, sample_output)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lisa_dataset_dir', default='../../data/lisa-cropped-dataset')
    parser.add_argument('--bosch_dataset_dir', default='../../data/bosch-cropped-dataset')
    parser.add_argument('--dtld_dataset_dir', default='../../data/dtld-cropped-dataset')
    parser.add_argument('--output_dir', default='../../data/training-dataset')
    return parser.parse_args()


def main(args):
    copy_images(Path(args.lisa_dataset_dir)
                , Path(args.bosch_dataset_dir)
                , Path(args.dtld_dataset_dir)
                , Path(args.output_dir))


if __name__ == '__main__':
    main(parse_args())
