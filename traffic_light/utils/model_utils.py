import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


# Flatten 'x' to a single dimension, often used at the end of a model. 'full' for rank-1 tensor"
class Flatten(nn.Module):
    def __init__(self, full: bool = False):
        super().__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class AdaptiveConcatPool2d(nn.Module):
    # Layer that concatenates 'AdaptiveAvgPool2d' and 'AdaptiveMaxPool2d'
    def __init__(self, sz: Optional[int] = None):
        # Output will be 2*sz or 2 if sz is None
        super().__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


# Custom Head from FastAI library
def fast_ai_custom_head(number_of_features, number_of_classes):
    return nn.Sequential(
        AdaptiveConcatPool2d(),
        Flatten(),
        nn.BatchNorm1d(number_of_features),
        nn.Dropout(p=0.375),
        nn.Linear(number_of_features, 512),
        nn.ReLU(True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.75),
        nn.Linear(512, number_of_classes),
    )


def load_fastai_based_model(path, number_of_classes, model):
    # Loading checkpoint (file) from path
    fastai_model = torch.load(path)

    # Architecture selection based on 'path'
    # If path does not specify, use model from parameters
    if 'resnet-34' in path:
        number_of_features = 1024
        model = models.resnet34()
    elif 'resnet-50' in path:
        number_of_features = 4096
        model = models.resnet50()
    elif 'densenet-121' in path:
        number_of_features = 2048
        model = models.densenet121()
    else:
        model = model

    # Modification of modules in state dictionary to match fastai architectures
    modules = list(model.children())

    if 'resnet' in path:  # Need to pop last two extra layers to match fastai architecture
        modules.pop(-1)
        modules.pop(-1)
    elif 'densenet' in path:  # Need to pop last layer to match fastai architecture
        modules.pop(-1)

    temp = nn.Sequential(nn.Sequential(*modules))
    temp_children = list(temp.children())
    temp_children.append(fast_ai_custom_head(number_of_features, number_of_classes))

    model = nn.Sequential(*temp_children)

    if 'pickle' in path:  # If pickle file
        model.load_state_dict(fastai_model['model'].state_dict())
    elif 'state_dict' in path:  # If state_dict file
        model.load_state_dict(fastai_model['state_dict'])

    return model.cuda()


def get_labels():
    return ['go', 'goLeft', 'stop', 'stopLeft']


if __name__ == '__main__':
    print('This class is only callable, not executable')
