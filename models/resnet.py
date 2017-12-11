import torch.nn as nn
import math
import torchvision.models as model_zoo

def resnet34(pretrained=False, num_langs=2, **kwargs):

    conv2d = nn.Conv2d(1, 3, 1) # turn 1 channel into 3 to simulate image
    conv2d.weight.data[0] = 1. # ensure original spectrogram is maintained

    resnet = model_zoo.resnet34(pretrained=pretrained, **kwargs)
    # change the last fc layer
    resnet.fc = nn.Linear(512 * 1, num_langs)

    model = nn.Sequential(conv2d, resnet)

    return model

def resnet101(pretrained=False, num_langs=2, **kwargs):

    conv2d = nn.Conv2d(1, 3, 1) # turn 1 channel into 3 to simulate image
    conv2d.weight.data[0] = 1. # ensure original spectrogram is maintained

    resnet = model_zoo.resnet101(pretrained=pretrained, **kwargs)
    # change the last fc layer

    resnet.fc = nn.Linear(2048 * 1, num_langs)

    model = nn.Sequential(conv2d, resnet)

    return model
