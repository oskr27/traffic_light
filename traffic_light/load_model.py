import torch
import torch.nn as nn
from torchvision import models
from traffic_light import model_utils


def load_fastai_based_model(path, number_of_classes, model=nn.Module):
    # Loading checkpoint (file) from path
    fastai_model = torch.load(path)

    # Architecture selection based on 'path'
    # If path does not specify, use model from parameters
    if 'resnet-34' in path:
        model = models.resnet34()
    elif 'resnet-50' in path:
        model = models.resnet50()
    elif 'densenet-121' in path:
        model = models.densenet121()
    else:
        model = model

    # Module elimination from standard models in nn.Models
    modules = list(model.children())
    modules.pop(-1)
    modules.pop(-1)

    temp = nn.Sequential(nn.Sequential(*modules))
    temp_children = list(temp.children())
    temp_children.append(model_utils.fast_ai_resnet34_head(1024, number_of_classes))

    model = nn.Sequential(*temp_children)

    if 'pickle' in path:  # If pickle file
        model.load_state_dict(fastai_model['model'].state_dict())
    elif 'state_dict' in path:  # If state_dict file
        model.load_state_dict(fastai_model['state_dict'])

    return model.cuda()


if __name__ == '__main__':
    print('This module can only be imported')
