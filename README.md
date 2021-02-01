# squeezenas_train

## Introduction:
This is a training script for the [SqueezeNAS](https://github.com/ashaw596/squeezenas) models
on the [RailSem19](https://wilddash.cc/railsem19).




## How to start with the script:
In order to start with the script you have to first download the **SqueezeNAS** repository and then download and exctract this repository in the same folder where you have your **SqueezeNAS** scripts. There is a helper repository [HERE](https://github.com/themozel/segmentation_models_pytorch.git) which is also needed to be downloaded into the same directory. Our script is written specifically for **RailSem** so you can also download this from the provided link into the directory and start with the training.


## Training:
Using `rs_train.py` you can train your models. By default the training is performed on the pretrained (on CityScapes dataset) weights of SqueezeNAS models which are located in the `weights` directory of that repository.

## Inference:
1. Latency: We are measuring the inference time in `rs_latency.py` using cuda events.
2. IoU-Score: Using `rs_iou_visual.py` you can calculate the IoU-Values of the predictions and also visualize the results.

## Results:
This is our results for evaluation of our trained models on RailSem19 dataset:

|     Architecture      |   mIOU   | Latency Values(ms) |
| -------------         | -------- |    -------------   |
| SqueezeNAS MAC Small  |   36.62  |       34.36        |
| SqueezeNAS MAC Large  |   41.48  |       76.23        |
| SqueezeNAS MAC XLarge |   44     |       178.78       |
| SqueezeNAS LAT Small  |   40.49  |       39.79        |
| SqueezeNAS LAT Large  |   42.12  |       116.68       |
| SqueezeNAS LAT XLarge |   46.76  |       180.18       |





 





















