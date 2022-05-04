import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as trn
from tqdm import tqdm

from VCD import models
from VCD.datasets.dataset import ListDataset
from VCD.utils import DEVICE_STATUS, DEVICE_COUNT


@torch.no_grad()
def extract_frame_features(model, loader, data_root, save_to):
    model.eval()
    videos, frames = [], []
    videos_fc = []

    data_root = os.path.join(data_root, 'frames')
    for video_class in os.listdir(data_root):
        for video in os.listdir(os.path.join(data_root, video_class)):
            videos.append(os.path.join(video_class, video))
            frame_list = os.listdir(os.path.join(data_root, video_class, video))
            for frame in frame_list:
                frames.append(os.path.join(data_root, video_class, video, frame))
            videos_fc.append(len(frame_list))

    loader.dataset.l = frames
    bar = tqdm(loader, ncols=150, unit='batch')
    features = []

    vidx = 0
    for idx, (paths, frames) in enumerate(bar):
        feat = model(frames.cuda()).cpu()
        features.append(feat)

        features = torch.cat(features)
        while vidx < len(videos):
            c = videos_fc[vidx]
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
    parser = argparse.ArgumentParser(description='Extract frame feature')
    parser.add_argument('--model', default='Resnet26_ViTS32_in21k')
    parser.add_argument('--ckpt', type=str, default='/workspace/ckpt/res26_vits32_fivr_triplet.pth')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--feature_path', type=str, default='/mldisk/nfs_shared_/sy/sea_story/features')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--worker', type=int, default=8)
    args = parser.parse_args()

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

    loader = DataLoader(ListDataset([], transform=transform), batch_size=args.batch, shuffle=False,
                        num_workers=args.worker)

    state_dict = torch.load(args.ckpt, map_location='cpu')['state_dict']
    state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    extract_frame_features(model, loader, args.dataset_root, args.feature_path)
