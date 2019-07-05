from pathlib import Path
import random


def main():
    master_path = Path('../../data/')
    path_input = master_path/'cropped-dataset-no-filter'

    folders = []
    for folder in path_input.glob('*/*'):
        folders.append(folder)

    images = []
    for folder in folders:
        aux = []
        for file in folder.iterdir():
            aux.append(file)

        selected_images = random.sample(aux, k=int(len(aux)*0.1))
        images += selected_images
        print(len(selected_images))

    print(len(images))
    #selected_images = random.sample(images, k=int(len(images)*0.1))
    #print(len(images))
    #print(len(selected_images))


if __name__ == '__main__':
    main()
