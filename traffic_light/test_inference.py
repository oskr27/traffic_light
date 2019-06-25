import torch
import torch.nn as nn
from torchvision import transforms, models
from pathlib import Path
from PIL import Image


def load_image_as_tensor(path):
    image = Image.open('../data/training-dataset/inference/dayClip1_46.jpg')
    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze_(0)
    return tensor_image.cuda()


def main():
    exported_models_path = Path('../data/training-dataset/models/pickles')
    model_list = [file for file in list(exported_models_path.glob('*.pkl'))]

    exported_models_path = Path('../data/training-dataset/models')
    model_list = [file for file in list(exported_models_path.glob('*.pth'))]

    sample_image = load_image_as_tensor('../data/training-dataset/inference/dayClip1_46.jpg')

    checkpoint = torch.load(model_list[0], map_location='cuda')
    model = models.resnet34(pretrained=False)
    print('\nBefore loading...')
    print(model.state_dict())
    model.cuda()
    model.load_state_dict(checkpoint['model'], strict=False)
    # print(model.state_dict())

    model.eval()
    print('\nAfter loading...')
    print(model.state_dict())
    out_pred = model(sample_image)
    print(out_pred)
    print(out_pred.size())

    pred = out_pred.max(1, keepdim=True)[1]
    print(pred.data[0])


if __name__ == '__main__':
    main()
