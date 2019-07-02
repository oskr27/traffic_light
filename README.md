# Traffic Light Detector #

Welcome to the Traffic Light Detector Repository by Oscar ROSAS (PF Lab @ The University of Tokyo)


## How to install
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
$ conda activate base && conda update conda
$ git clone https://github.com/oskr27/traffic_light
$ cd traffic_light && conda env create -f environment.yml
$ conda activate traffic
$ cd traffic_lights
$ python test_inference.py '../data/training-dataset/inference/dayClip1_154.jpg'
```

Output:
```
'../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-34-no_tuning_dict.pth','go',0.950,0.166[ms]
'../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-50-no_tuning_dict.pth','go',0.963,0.00719[ms]
'../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/densenet-121-no_tuning_dict.pth','go',0.915,0.0164[ms]
'../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-34-tuned_dict.pth','go',0.996,0.00518[ms]
'../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-50-tuned_dict.pth','go',0.959,0.00658[ms]
'../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/densenet-121-tuned_dict.pth','go',0.937,0.0126[ms]
```

Format:
`Image_path, Model_path, inference_result, class_score, inference_time`