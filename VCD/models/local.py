from collections import OrderedDict
from torchvision import models

import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

from VCD.models.pooling import L2N, GeM, RMAC
from VCD.models.frame import BaseModel


class MobileNet_local(BaseModel):
    def __init__(self):
        super(MobileNet_local, self).__init__()
        self.base = nn.Sequential(
            OrderedDict([*list(models.mobilenet_v2(pretrained=True).features.named_children())])
        )
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.norm(x)
        x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[2]).squeeze(-1)
        return x


class Resnet50_local(BaseModel):
    def __init__(self):
        super(Resnet50_local, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.norm(x)
        x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[2]).squeeze(-1)
        return x


class Resnet50_intermediate(BaseModel):
    def __init__(self):
        super(Resnet50_intermediate, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-4]
                                              + [('layer3', nn.Sequential(
            list(models.resnet50(pretrained=True).named_children())[6][1][:5]))]))  # conv4_5

        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.norm(x)
        x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[2]).squeeze(-1)
        return x

class Local_Maxpooling(BaseModel):
    def __init__(self, group_count):
        super(Local_Maxpooling, self).__init__()
        self.pool = torch.nn.MaxPool1d(group_count)
        self.norm = L2N()

    def forward(self, x):
        x = x.permute(1,2,0)
        x = self.pool(x)
        x = x.permute(2, 0, 1)
        x = self.norm(x)
        return x
