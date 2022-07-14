import argparse
import os

import torch
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Multi-Modal Feature')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--mode', type=str, default='concat', choices=['sum', 'concat'])
    parser.add_argument('--feature', type=str, default='MFCC', choices=['MFCC'])

    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.dataset_root, 'videos'))
    feat_dir = os.path.abspath(os.path.join(args.dataset_root, f'features_{args.duration}s'))
    audio_feat_dir = os.path.abspath(os.path.join(args.dataset_root, f'audio_features_{args.feature}_{args.duration}s'))
    multi_feat_dir = os.path.abspath(
        os.path.join(args.dataset_root, f'multi_features_{args.feature}_{args.duration}s_{args.mode}'))

    video_cls = ['HighLight', 'Origin']
    videos = sorted([os.path.join(c, v) for c in video_cls for v in os.listdir(os.path.join(video_dir, c))])

    if os.path.exists(multi_feat_dir):
        print(f'Multi Modal Feature directory {multi_feat_dir} is already exist')
        exit(1)

    for c in video_cls:
        os.makedirs(os.path.join(multi_feat_dir, c))

    bar = tqdm(videos, ncols=150, unit='video')

    for v in bar:
        vision_feat = torch.load(os.path.join(feat_dir, f'{v}.pth'))
        audio_feat = torch.load(os.path.join(audio_feat_dir, f'{v}.pth'))

        if vision_feat.shape[0] < audio_feat.shape[0]:
            audio_feat = audio_feat[:-1]
        elif vision_feat.shape[0] > audio_feat.shape[0]:
            vision_feat = vision_feat[:-1]

        multi_feat = None
        if args.mode == 'sum':
            multi_feat = 0.9 * vision_feat + 0.1 * audio_feat
        elif args.mode == 'concat':
            multi_feat = torch.cat((vision_feat, audio_feat), 1)
        torch.save(multi_feat, os.path.join(multi_feat_dir, f'{v}.pth'))
