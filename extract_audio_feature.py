import argparse
import os
import warnings

warnings.filterwarnings(action='ignore')

from tqdm import tqdm
import torch
from torchvision.transforms import transforms as trn

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from PIL import Image
import joblib
from VCD.utils import DEVICE_STATUS, DEVICE_COUNT
from VCD import models


class MelDataset(Dataset):
    def __init__(self, mel, transform=None):
        self.mel = mel
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

        if isinstance(self.transform, trn.Compose):
            self.load = lambda x: self.transform(default_loader(x))
        else:
            raise TypeError('Unsupported image loader')

    def __getitem__(self, idx):
        img = self.mel[idx]
        img = np.concatenate((img, img, img), axis=0)
        img = Image.fromarray(np.uint8(img).transpose(1, 2, 0))
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.mel)


def load_mels(mels_path):
    full_mels = np.load(mels_path)
    segment_mels = np.split(full_mels, full_mels.shape[0], axis=0)
    return np.array(segment_mels)


@torch.no_grad()
def extract_audio_features(model, loader, videos, videos_sc, save_to):
    model.eval()
    features = []
    bar = tqdm(loader, ncols=150, unit='batch')

    vidx = 0
    for idx, imgs in enumerate(bar):
        feat = model(imgs.cuda()).cpu()
        features.append(feat)

        features = torch.cat(features)
        while vidx < len(videos):
            c = videos_sc[vidx]
            if features.shape[0] >= c:
                target = os.path.join(save_to, f'{videos[vidx]}.pth')
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                torch.save(features[:c, ], target)
                bar.set_description_str(os.path.basename(target))
                features = features[c:, ]
                vidx += 1
            else:
                break
        features = [features]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Segment Audio Feature')
    parser.add_argument('--model', default='Resnet26_ViTS32_in21k')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--feature', type=str, default='MelSpectogram', choices=['ConstantQ', 'MelSpectogram'])

    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.dataset_root, 'videos'))
    feature_dir = os.path.abspath(os.path.join(args.dataset_root, f'{args.feature}_{args.duration}s'))
    audio_feat_dir = os.path.abspath(os.path.join(args.dataset_root, f'audio_features_{args.feature}_{args.duration}s'))

    video_cls = ['HighLight', 'Origin']
    videos = sorted([os.path.join(c, v) for c in video_cls for v in os.listdir(os.path.join(video_dir, c))])

    # Resnet26_ViTS32_in21k
    model = models.get_frame_model(args.model).cuda()

    # Check device
    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    transform = trn.Compose([
        trn.Resize((224, 224)),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pool = joblib.Parallel(args.worker)
    mapper = joblib.delayed(load_mels)
    tasks = [mapper(os.path.join(args.dataset_root, feature_dir, f'{v}.npy')) for v in videos]
    videos_segment = pool(tqdm(tasks))

    videos_sc = []
    audio_image_store = np.array([])
    for one_video_segment in videos_segment:
        if len(audio_image_store) == 0:
            audio_image_store = one_video_segment
        else:
            audio_image_store = np.concatenate((audio_image_store, one_video_segment), axis=0)
        videos_sc.append(one_video_segment.shape[0])

    loader = DataLoader(MelDataset(audio_image_store, transform=transform), batch_size=args.batch, shuffle=False,
                        num_workers=args.worker)

    extract_audio_features(model, loader, videos, videos_sc, audio_feat_dir)
