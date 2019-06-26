import torch
import torch.nn as nn
from typing import Optional


# Flatten 'x' to a single dimension, often used at the end of a model. 'full' for rank-1 tensor"
class Flatten(nn.Module):
    def __init__(self, full:bool=False):
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
def fast_ai_resnet34_head(number_of_features, number_of_classes):
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


def main():
    print('This class is only callable, not executable')


if __name__ == '__main__':
    main()
