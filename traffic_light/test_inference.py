from pathlib import Path
from PIL import Image
from torchvision import transforms
from traffic_light import model


def main():
    models_path = Path('../data/training-dataset/models')
    sample = Image.open('../data/training-dataset/inference/dayClip1_46.jpg')
    tensor_sample = transforms.ToTensor()(sample)

    print(list(models_path.glob('**/*.pth')))
    print(tensor_sample)

    my_model = model.load_model(models_path/'resnet-34-no_tuning.pth')
    my_model.eval()

    out_predict = my_model(tensor_sample)

    print(out_predict)
    pred = out_predict.max(1, keepdim=True)[1]
    print(pred)


if __name__ == '__main__':
    main()
