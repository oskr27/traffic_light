import torch
import matplotlib.pyplot as plt
from torchvision import transforms, utils


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        height, width = image.shape[:2]

        if isinstance(self.output_size, int):
            if height > width:
                new_height, new_width = self.output_size * height / width, self.output_size
            else:
                new_height, new_width = self.output_size, self.output_size * width / height
        else:
            new_height, new_width = self.output_size

        new_height, new_width = int(new_height), int(new_width)

        img = transforms.resize(image, (new_height, new_width))
        return img


class ToTensor(object):
    def __call__(self, sample):
        image, = sample

        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


return transforms.ToTensor