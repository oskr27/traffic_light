import torch
import torch.nn as nn
from torchvision import transforms, models
import load_model
import cv2
import time
import argparse
import os

def load_image_as_tensor(path):
    cv_image = cv2.imread(path, 1)
    image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.unsqueeze_(0)
    return tensor_image.cuda(), cv_image


def infer_image(image_path, model_path):
    number_of_classes = 4
    tensor_image, cv_image = load_image_as_tensor(image_path)
    model = load_model.load_fastai_based_model(model_path, number_of_classes, None)

    model.eval()

    soft_max = torch.nn.Softmax(dim=1)
    t = time.process_time()
    raw_out = model(tensor_image)
    out = soft_max(raw_out)
    elapsed_time = time.process_time() - t
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

    print("'{:}','{:}','{:}',{:.3f},{:.3}[ms]".format(image_path, model_path, tags[max_index], max_prob, elapsed_time))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv_image, tags[max_index], (10, 20), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("{:}".format(os.path.basename(model_path)),cv_image)
    cv2.waitKey(0)

def main():
    parser = argparse.ArgumentParser(description='Traffic Light inference ')
    parser.add_argument('image', help='Image to test', type=str, default='../data/training-dataset/test/dayClip1_46.jpg')
    args = parser.parse_args()

    image_path = args.image

    model_path = '../data/training-dataset/models/state_dict/resnet-34-no_tuning_dict.pth'
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
    main()
