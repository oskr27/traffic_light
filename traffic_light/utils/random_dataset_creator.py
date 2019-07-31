from pathlib import Path
import random
import argparse
import os
import shutil


def ir(some_value):
    return int(round(some_value))


def is_selected(probability):
    return random.random() < probability


def copy_images(lisa_dataset_dir, bosch_dataset_dir, dtld_dataset_dir, output_dir, generate_sample):
    # Cleaning output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if generate_sample and os.path.exists(str(output_dir) + '-sample'):
        shutil.rmtree(str(output_dir) + '-sample')

    images = []

    lisa_modifier = 0.1
    bosch_modifier = 1.0
    dtld_modifier = 1.0

    if lisa_dataset_dir != '':
        lisa_images = [file for file in lisa_dataset_dir.glob('**/*.*')]
        images += random.sample(lisa_images, ir(len(lisa_images) * lisa_modifier))
    if bosch_dataset_dir != '':
        bosch_images = [file for file in bosch_dataset_dir.glob('**/*.*')]
        images += random.sample(bosch_images, ir(len(bosch_images) * bosch_modifier))
    if dtld_dataset_dir != '':
        dtld_images = [file for file in dtld_dataset_dir.glob('**/*.*') if 'Right' not in str(file.parent)]
        images += random.sample(dtld_images, ir(len(dtld_images) * dtld_modifier))

    # Then, we need to create the dataset using the six classes go, goLeft, stop, stopLeft, warning, and warningLeft
    # For this, we need to have a base of around 1000 images per class (2000 would be ideal)
    for image in images:
        if str(lisa_dataset_dir) in str(image):
            output = str.replace(str(image), str(lisa_dataset_dir), str(output_dir))
        elif str(bosch_dataset_dir) in str(image):
            output = str.replace(str(image), str(bosch_dataset_dir), str(output_dir))
        elif str(dtld_dataset_dir) in str(image):
            output = str.replace(str(image), str(dtld_dataset_dir), str(output_dir))

        # Merging warning to stop
        if not os.path.exists(Path(output).parent):
            os.makedirs(Path(output).parent)

        shutil.copy(image, output)

        if is_selected(0.01) and generate_sample:
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
    parser.add_argument('--generate_sample', default=False)
    return parser.parse_args()


def main(args):
    copy_images(Path(args.lisa_dataset_dir)
                , Path(args.bosch_dataset_dir)
                , Path(args.dtld_dataset_dir)
                , Path(args.output_dir)
                , args.generate_sample)


if __name__ == '__main__':
    main(parse_args())
