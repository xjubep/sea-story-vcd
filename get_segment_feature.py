import argparse
import os

import torch
from tqdm import tqdm

from VCD import models
from VCD.utils import DEVICE_STATUS, DEVICE_COUNT

FEATURE_EXTENSION = ['pt', 'pth']


@torch.no_grad()
def extract_segment_features(model, count, feature_dir, save_to):
    model.eval()
    feature_path = [os.path.join(r, feat) for r, d, f in os.walk(feature_dir) for feat in f if
                    feat.split('.')[-1].lower() in FEATURE_EXTENSION]

    for p in tqdm(feature_path, ncols=150, unit='video'):
        frame_feature = torch.load(p)

        k = count - frame_feature.shape[0] % count
        if k != count:
            frame_feature = torch.cat([frame_feature, frame_feature[-1:, ].repeat((k, 1))])
        frame_feature = frame_feature.reshape(-1, count, frame_feature.shape[-1])
        segment_feature = model(frame_feature.cuda()).cpu()
        target = os.path.join(save_to, os.path.relpath(p, feature_dir))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        torch.save(segment_feature, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract segment feature')
    parser.add_argument('--model', type=str, default='Segment_Tformer_MaxPool')  # #head=1, #layer=1
    parser.add_argument('--frame_feature_path', type=str, default='/mldisk/nfs_shared_/sy/sea_story/features')
    parser.add_argument('--segment_feature_path', type=str, default='/mldisk/nfs_shared_/sy/sea_story/features_5s')
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.frame_feature_path):
        print(f'Frame feature path {args.frame_feature_path} does not exist.')
        exit(1)
    if os.path.exists(args.segment_feature_path) and len(os.listdir(args.segment_feature_path)) != 0:
        print(f'Segment feature path {args.segment_feature_path} is not clean.')
        exit(1)

    # models
    model = models.get_segment_model(args.model).cuda()

    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    extract_segment_features(model, args.count, args.frame_feature_path, args.segment_feature_path)
