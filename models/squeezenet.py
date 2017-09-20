import torch.nn as nn
import math
import torchvision.models as model_zoo

def squeezenet(pretrained=False, num_langs=5, **kwargs):

    conv2d = nn.Conv2d(1, 3, 1) # turn 1 channel into 3 to simulate image
    conv2d.weight.data[0] = 1. # ensure original spectrogram is maintained 

    sqnet = model_zoo.squeezenet1_1(pretrained=pretrained, **kwargs)
    # change the last conv2d layer
    sqnet.classifier._modules["1"] = nn.Conv2d(512, num_langs, kernel_size=(1, 1))
    # change the internal num_classes variable rather than redefining the forward pass
    sqnet.num_classes = num_langs

    model = nn.Sequential(conv2d, sqnet)

    return model
