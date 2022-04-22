import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VGG16_bn_22333(nn.Module):
    def __init__(self,cfg):
        super(VGG16_bn_22333, self).__init__()
        self.net = models.vgg16_bn(pretrained=True)
        # self.net.classifier = nn.Sequential()
        # self.net.avgpool = nn.Sequential()
        self.features = self.net.features
        # for i in range(33, 44):
        #     del self.features._modules['{index}'.format(index=i)]
        del self.features._modules['{index}'.format(index=43)]

    def forward(self, x):
        x = self.features(x)
        return x