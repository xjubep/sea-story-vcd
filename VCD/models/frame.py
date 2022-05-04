from collections import OrderedDict

from torchvision import models
import torch.nn as nn
import torch

import timm

from VCD.models.pooling import L2N, GeM, RMAC
from VCD.models.summary import summary

FRAME_MODELS = ['MobileNet_AVG', 'Resnet50_RMAC', 'Resnet50_GeM', 'Effb4ns_RMAC', 'Effb3ns_RMAC', 'Resnet50_ViTB16_384',
                'Resnet26_ViTS32_in21k']


class BaseModel(nn.Module):
    def __str__(self):
        return self.__class__.__name__

    def summary(self, input_size, batch_size=-1, device="cuda"):
        try:
            return summary(self, input_size, batch_size, device)
        except:
            return self.__repr__()


class MobileNet_AVG(BaseModel):
    def __init__(self):
        super(MobileNet_AVG, self).__init__()
        self.base = nn.Sequential(OrderedDict(models.mobilenet_v2(pretrained=True).features.named_children()))
        # self.base = nn.Sequential(OrderedDict(models.mobilenet_v2(pretrained=True).named_children()))

        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x


class Resnet50_RMAC(BaseModel):
    def __init__(self):
        super(Resnet50_RMAC, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x

class Resnet50_GeM(BaseModel):
    def __init__(self):
        super(Resnet50_GeM, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(models.resnet50(pretrained=True).named_children())[:-2]))
        self.pool = GeM()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.norm(x)
        return x

class Effb4ns_RMAC(BaseModel):
    def __init__(self):
        super(Effb4ns_RMAC, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(timm.create_model('tf_efficientnet_b4_ns', pretrained=True).named_children())[:-2]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x

class Effb3ns_RMAC(BaseModel):
    def __init__(self):
        super(Effb3ns_RMAC, self).__init__()
        self.base = nn.Sequential(OrderedDict(list(timm.create_model('tf_efficientnet_b3_ns', pretrained=True).named_children())[:-2]))
        self.pool = RMAC()
        self.norm = L2N()

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x)
        x = self.norm(x)
        return x

class Resnet26_ViTS32_in21k(BaseModel):
    def __init__(self):
        super(Resnet26_ViTS32_in21k, self).__init__()
        self.base = timm.create_model('vit_small_r26_s32_224_in21k', pretrained=True)

    def forward(self, x):
        self.base.head = nn.Identity()
        x = self.base(x)
        return x

if __name__ == '__main__':
    # model = Resnet50_RMAC() # #para: 23.5M, Total Size: 376.82 MB 80.372%
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model.__repr__())
    # # print(model)
    #
    # model = MobileNet_AVG() # #para: 23.5M, Total Size: 376.82 MB 80.372%
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model.__repr__())
    # # print(model)

    # model = Resnet50_GeM() # #para: 23.5M, Total Size: 376.82 MB 80.372%
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model.__repr__())

    # model = Effb4ns_RMAC()  # #para: 17.5M, Total Size: 513.29 MB 85.162%
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model)

    # model = Effb3ns_RMAC()  # 1536dim, #para: 10.7M, Total Size: 380.79 MB 84.048%
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model)

    # model = Resnet50_ViTB16_384()  # 768dim, #para: 97.7M, Total Size: 1207.67 MB 84.972%
    # print(model.summary((3, 224, 224), device='cpu'))
    # print(model)

    model = Resnet26_ViTS32_in21k()  # 384dim, #para: 36.0M, Total Size: 452.47 MB x%
    print(model.summary((3, 224, 224), device='cpu'))
    print(model.__repr__())
    print(model)

    # Total size = Input size + Forward/backward pass size + Params size