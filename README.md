# Traffic Light Detector #
Welcome to the Traffic Light Detector Repository by Oscar ROSAS (PF Lab @ The University of Tokyo)

## Getting Started ##
The main components for you to get the project up and running are the ones below:
* Anaconda 3
* Python 3

In addition, the project assumes that you have the following configuration on your machine:
* A Linux-based OS distribution
* An Nvidia architecture GPU device

### How to Install ###
Using the terminal, execute the following instructions
1. Install Conda Virtual Environment Manager (the example installs Miniconda, but you can use Anaconda as well)

    ```
    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh
    $ conda activate base && conda update conda
    ```

1. Clone this repository 

    ```
    $ git clone https://github.com/oskr27/traffic_light traffic_light
    ```

1. Create the virtual environment from the YML file in the repository

    ```
    $ cd traffic_light && conda env create -f traffic_light/environment.yml
    ```

1. Activate the virtual environment, and you should be all set
    
    ```
    $ conda activate traffic
    ```
    
### Testing a single image ###

After you cloned the repository and ensuring all the dependencies are met, you may want to infer the class of a single
image. For this, execute the instructions below in a terminal. 
    
    
    $ cd traffic_lights
    $ python test_inference.py '../data/training-dataset/inference/dayClip1_154.jpg'
    

Output:

    
    '../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-34-no_tuning_dict.pth','go',0.950,0.166[ms]
    '../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-50-no_tuning_dict.pth','go',0.963,0.00719[ms]
    '../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/densenet-121-no_tuning_dict.pth','go',0.915,0.0164[ms]
    '../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-34-tuned_dict.pth','go',0.996,0.00518[ms]
    '../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/resnet-50-tuned_dict.pth','go',0.959,0.00658[ms]
    '../data/training-dataset/inference/dayClip1_154.jpg','../data/training-dataset/models/state_dict/densenet-121-tuned_dict.pth','go',0.937,0.0126[ms]
    
    


Format:
`Image_path, Model_path, inference_result, class_score, inference_time`
