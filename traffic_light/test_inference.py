import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from traffic_light import load_model


def load_image_as_tensor(path):
    image = Image.open(path)
    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze_(0)
    return tensor_image.cuda()


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
    print(out[0])


if __name__ == '__main__':
    main()
