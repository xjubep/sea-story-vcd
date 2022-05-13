import argparse
import os

import torch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Multi-Modal Feature')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--duration', type=int, default=10)
    parser.add_argument('--worker', type=int, default=8)
    parser.add_argument('--mode', required=False, default='sum', help="[sum/concat] element_sum or concat")
    parser.add_argument('--feature', type=str, default='MelSpectogram', choices=['ConstantQ', 'MelSpectogram'])

    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.dataset_root, 'videos'))
    feat_dir = os.path.abspath(os.path.join(args.dataset_root, f'features_{args.duration}s'))
    audio_feat_dir = os.path.abspath(os.path.join(args.dataset_root, f'audio_features_{args.feature}_{args.duration}s'))
    multi_feat_dir = os.path.abspath(os.path.join(args.dataset_root, f'multi_features_{args.feature}_{args.duration}s_{args.mode}8_2'))

    video_cls = ['HighLight', 'Origin']
    videos = sorted([os.path.join(c, v) for c in video_cls for v in os.listdir(os.path.join(video_dir, c))])

    if os.path.exists(multi_feat_dir):
        print(f'Multi Modal Feature directory {multi_feat_dir} is already exist')
        exit(1)

    for c in video_cls:
        os.makedirs(os.path.join(multi_feat_dir, c))

    bar = tqdm(videos, ncols=150, unit='batch')

    for v in bar:
        vision_feat = torch.load(os.path.join(feat_dir, f'{v}.pth'))
        audio_feat = torch.load(os.path.join(audio_feat_dir, f'{v}.pth'))
        print(vision_feat.shape)
        print(audio_feat.shape)
        if args.mode == 'sum':
            multi_feat = 0.8 * vision_feat + 0.2 * audio_feat
        elif args.mode == 'concat':
            multi_feat = torch.cat((vision_feat, audio_feat), 1)
        torch.save(multi_feat, os.path.join(multi_feat_dir, f'{v}.pth'))
