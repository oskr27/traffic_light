from fastai.vision import *
from fastai.metrics import error_rate
from fastai.callbacks.mem import PeakMemMetric
from pathlib import Path

bs = 64
master_path = Path('../../data')
training_dataset_path = master_path/'training-dataset'
models_path = training_dataset_path / 'models'

metrics = [error_rate, accuracy]
callback_fns = PeakMemMetric


def export_pkl(path, model):
    if 'resnet-34' in model:
        arch = models.resnet34
    elif 'resnet-50' in model:
        arch = models.resnet50
    elif 'densenet-121' in model:
        arch = models.densenet121
    else:
        arch = None

    data_set = ImageDataBunch.from_folder(training_dataset_path, ds_tfms=get_transforms(), size=224, bs=bs)
    learn = cnn_learner(data_set, base_arch=arch).load(model)
    learn.export(path/Path(model + '.pkl'))


def export_state_dictionary(path, model):
    if 'resnet-34' in model:
        arch = models.resnet34
    elif 'resnet-50' in model:
        arch = models.resnet50
    elif 'densenet-121' in model:
        arch = models.densenet121
    else:
        arch = None

    data_set = ImageDataBunch.from_folder(training_dataset_path, ds_tfms=get_transforms(), size=224, bs=bs)
    learn = cnn_learner(data_set, base_arch=arch).load(model)
    torch.save({'state_dict': learn.model.state_dict()}, path/Path(model + '.pth'))


def export_h5(model):
    if 'resnet-34' in model:
        arch = models.resnet34
    elif 'resnet-50' in model:
        arch = models.resnet50
    elif 'densenet-121' in model:
        arch = models.densenet121
    else:
        arch = None

    data_set = ImageDataBunch.from_folder(training_dataset_path, ds_tfms=get_transforms(), size=224, bs=bs)
    learn = cnn_learner(data_set, base_arch=arch).load(model)
    learn.save(model + '.h5')


def main():
    model_list = [model for model in list(models_path.glob('*.pth'))]
    for file in model_list:
        if file.stem != 'tmp':
            print('Exporting ' + file.stem + '...')
            # export_state_dictionary(models_path / 'state_dict', file.stem)
            # export_pkl(models_path / 'pickle', file.stem)
            export_h5(models_path / 'h5', file.stem)


if __name__ == '__main__':
    main()
