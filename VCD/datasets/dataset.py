import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets.folder import default_loader
from torchvision.transforms import transforms as trn
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2


def albumentations_loader(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


class ListDataset(Dataset):
    def __init__(self, l, transform=None):
        self.l = l
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

        if isinstance(self.transform, A.Compose):
            self.load = lambda x: self.transform(image=albumentations_loader(x))['image']
        elif isinstance(self.transform, trn.Compose):
            self.load = lambda x: self.transform(default_loader(x))
        else:
            raise TypeError('Unsupported image loader')

    def __getitem__(self, idx):
        path = self.l[idx]
        frame = self.load(path)

        return path, frame

    def __len__(self):
        return len(self.l)

    def __repr__(self):
        fmt_str = f'{self.__class__.__name__}\n'
        fmt_str += f'\tNumber of images : {self.__len__()}\n'
        trn_str = self.transform.__repr__().replace('\n', '\n\t')
        fmt_str += f"\tTransform : \n\t{trn_str}"
        return fmt_str


class TripletDataset(Dataset):
    def __init__(self, triplets_csv, frame_root, transform=None):
        self.root = frame_root
        self.triplets = pd.read_csv(triplets_csv).values

        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

        if isinstance(self.transform, A.Compose):
            self.load = lambda x: self.transform(image=albumentations_loader(x))['image']
        elif isinstance(self.transform, trn.Compose):
            self.load = lambda x: self.transform(default_loader(x))
        else:
            raise TypeError('Unsupported image loader')

    def load_image(self, p):
        path = os.path.join(self.root, p)
        im = self.load(path)

        return path, im

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]

        anc_path, anc = self.load_image(a)
        pos_path, pos = self.load_image(p)
        neg_path, neg = self.load_image(n)

        return (anc_path, pos_path, neg_path), (anc, pos, neg)

    def __len__(self):
        return len(self.triplets)

