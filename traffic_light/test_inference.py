import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import time
import argparse
import os
from traffic_light.utils import model_utils


def load_image_as_tensor(path):
    cv_image = cv2.imread(path, 1)
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze_(0)
    return tensor_image.cuda(), cv_image


def infer_image(image_path, model_path):
    labels = model_utils.get_labels()
    number_of_classes = len(labels)

    tensor_image, cv_image = load_image_as_tensor(image_path)
    model = model_utils.load_fastai_based_model(model_path, number_of_classes, None)

    model.eval()

    soft_max = torch.nn.Softmax(dim=1)
    t = time.process_time()
    raw_out = model(tensor_image)
    out = soft_max(raw_out)
    elapsed_time = time.process_time() - t
    i = 0

    max_prob = 0
    max_index = 0

    for prob in out[0]:
        if prob.item() > max_prob:
            max_prob = prob.item()
            max_index = i
        i = i + 1

    print("'{:}','{:}','{:}',{:.3f},{:.3}[ms]".format(image_path, model_path, labels[max_index], max_prob, elapsed_time))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv_image, labels[max_index], (10, 20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("{:}".format(os.path.basename(model_path)), cv_image)
    cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser(description='Traffic Light Inference')
    parser.add_argument('--image', help='Image to be tested', default='../data/training-dataset/test/dayClip1_46.jpg')
    parser.add_argument('--model', help='A PTH Model', default='../data/training-dataset/models/state_dict/'
                                                               'resnet-34-no_tuning_dict.pth')
    return parser.parse_args()


def main(args):
    image_path = args.image
    model_path = args.model

    infer_image(image_path, model_path)

    model_path = '../data/training-dataset/models/state_dict/resnet-50-no_tuning_dict.pth'
    infer_image(image_path, model_path)

    model_path = '../data/training-dataset/models/state_dict/densenet-121-no_tuning_dict.pth'
    infer_image(image_path, model_path)

    model_path = '../data/training-dataset/models/state_dict/resnet-34-tuned_dict.pth'
    infer_image(image_path, model_path)

    model_path = '../data/training-dataset/models/state_dict/resnet-50-tuned_dict.pth'
    infer_image(image_path, model_path)

    model_path = '../data/training-dataset/models/state_dict/densenet-121-tuned_dict.pth'
    infer_image(image_path, model_path)


if __name__ == '__main__':
    main(parse_args())
