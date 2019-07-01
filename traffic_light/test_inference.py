import torch
import torch.nn as nn
from torchvision import transforms, models
from traffic_light import load_model
import cv2


def load_image_as_tensor(path):
    image = cv2.imread(path, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze_(0)
    return tensor_image.cuda()


def get_tags():
    tags = ['go', 'goLeft', 'stop', 'stopLeft']
    return tags


def main():
    model_path = '../data/training-dataset/models/state_dict/densenet-121-no_tuning_dict.pth'
    number_of_classes = 4
    model = load_model.load_fastai_based_model(model_path, number_of_classes, model=models.resnet34())

    sample_image = load_image_as_tensor('../data/training-dataset/inference/dayClip1_46.jpg')
    model.eval()

    print(sample_image.shape)
    raw_out = model(sample_image)

    soft_max = torch.nn.Softmax(dim=1)
    out = soft_max(raw_out)

    tags = get_tags()
    i = 0
    max_prob = 0
    max_index = 0
    for prob in out[0]:
        print(tags[i] + ': ' + str(prob))

        if prob > max_prob:
            max_prob = prob
            max_index = i
        i = i + 1

    print('Inference Result: ' + tags[max_indx] + ' (' + out[0][max_index] + ')')


if __name__ == '__main__':
    main()
