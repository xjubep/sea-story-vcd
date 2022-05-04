from VCD.models.frame import BaseModel
from VCD.models.pooling import L2N

import torch
import torch.nn as nn

SEGMENT_MODELS = ['Segment_MaxPool', 'Segment_AvgPool', 'Segment_Tformer_MaxPool', 'Segment_Tformer_AvgPool', 'Segment_Tformer']


class Segment_MaxPool(BaseModel):
    def __init__(self):
        super(Segment_MaxPool, self).__init__()
        self.norm = L2N()

    def forward(self, x):
        x, _ = torch.max(x, 1)
        x = self.norm(x)
        return x


class Segment_AvgPool(BaseModel):
    def __init__(self):
        super(Segment_AvgPool, self).__init__()
        self.norm = L2N()

    def forward(self, x):
        x, _ = torch.mean(x, 1)
        x = self.norm(x)
        return x

class Segment_Tformer_MaxPool(BaseModel):
    def __init__(self):
        super(Segment_Tformer_MaxPool, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.norm = L2N()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x, _ = torch.max(x, 1)
        x = self.norm(x)
        return x

class Segment_Tformer_AvgPool(BaseModel):
    def __init__(self):
        super(Segment_Tformer_AvgPool, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.norm = L2N()

    def forward(self, x):
        # x, _ = torch.mean(x, 1)
        x = self.transformer_encoder(x)
        x = torch.mean(x, 1)
        x = self.norm(x)
        return x

class Segment_Tformer(BaseModel):
    def __init__(self):
        super(Segment_Tformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=384, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.norm = L2N()

    def forward(self, x):
        x = self.transformer_encoder(x)
        # x = torch.mean(x, 1)
        x = self.norm(x)
        return x

if __name__ == '__main__':
    model = Segment_Tformer() # #para: 23.5M, Total Size: 376.82 MB 80.372%
    print(model.summary((6, 5, 384), device='cpu'))
    print(model.__repr__())
