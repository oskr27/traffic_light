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


def infer_image(image_path, model_path):
    number_of_classes = 4
    sample_image = load_image_as_tensor(image_path)
    model = load_model.load_fastai_based_model(model_path, number_of_classes, None)

    model.eval()

    soft_max = torch.nn.Softmax(dim=1)
    raw_out = model(sample_image)
    out = soft_max(raw_out)

    i = 0
    tags = ['go', 'goLeft', 'stop', 'stopLeft']
    max_prob = 0
    max_index = 0

    for prob in out[0]:
        # print('\t' + tags[i] + ': ' + str(prob.item() * 100))

        if prob.item() > max_prob:
            max_prob = prob.item()
            max_index = i
        i = i + 1

    print('  *** Inference Result: ' + tags[max_index] + ' with ' + str(max_prob * 100) + '%')


def main():
    print('Inference Test')

    image_path = '../data/training-dataset/inference/dayClip1_46.jpg'
    model_path = '../data/training-dataset/models/state_dict/resnet-34-no_tuning_dict.pth'
    print('Using model: ' + model_path)
    infer_image(image_path, model_path)

    image_path = '../data/training-dataset/inference/dayClip1_46.jpg'
    model_path = '../data/training-dataset/models/state_dict/resnet-50-no_tuning_dict.pth'
    print('Using model: ' + model_path)
    infer_image(image_path, model_path)

    image_path = '../data/training-dataset/inference/dayClip1_46.jpg'
    model_path = '../data/training-dataset/models/state_dict/densenet-121-no_tuning_dict.pth'
    print('Using model: ' + model_path)
    infer_image(image_path, model_path)

    image_path = '../data/training-dataset/inference/dayClip1_46.jpg'
    model_path = '../data/training-dataset/models/state_dict/resnet-34-tuned_dict.pth'
    print('Using model: ' + model_path)
    infer_image(image_path, model_path)

    image_path = '../data/training-dataset/inference/dayClip1_46.jpg'
    model_path = '../data/training-dataset/models/state_dict/resnet-50-tuned_dict.pth'
    print('Using model: ' + model_path)
    infer_image(image_path, model_path)

    image_path = '../data/training-dataset/inference/dayClip1_46.jpg'
    model_path = '../data/training-dataset/models/state_dict/densenet-121-tuned_dict.pth'
    print('Using model: ' + model_path)
    infer_image(image_path, model_path)


if __name__ == '__main__':
    main()
