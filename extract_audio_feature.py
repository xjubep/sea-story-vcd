import argparse
import os
import warnings

import torch

warnings.filterwarnings(action='ignore')

from tqdm import tqdm

import librosa as lb
import librosa.display as lbd
import numpy as np

import matplotlib.pyplot as plt


def create_mfcc(segment, config, win_length, to_db=True, norm=True):
    y, _ = lb.load(segment)
    s = lb.feature.melspectrogram(y=y, sr=config.sampling_rate, win_length=win_length, hop_length=config.hop_length,
                                  n_mels=config.n_mels)

    if (config.duration * config.sampling_rate) // config.hop_length + 2 - s.shape[1] > 0:
        s = np.pad(s, pad_width=(
            (0, 0), (0, (config.duration * config.sampling_rate) // config.hop_length + 2 - s.shape[1])),
                   mode='constant')

    if to_db:
        s = lb.power_to_db(s, ref=np.max)
    mfcc = lb.feature.mfcc(S=s, n_mfcc=config.n_mfcc)
    if norm:
        mfcc = mfcc / (np.linalg.norm(mfcc, axis=1, keepdims=True) + 1e-6)
    return mfcc


def create_3_mfccs(segment, config, mode='avg'):
    mfccs = [create_mfcc(segment, config, win_length=275 * (2 ** i)) for i in range(3)]

    if mode == 'avg':
        mfccs = [np.mean(mfcc, axis=1) for mfcc in mfccs]

    mfcc = np.concatenate(mfccs, axis=0)
    mfcc = mfcc.reshape(-1, mfcc.shape[0])
    mfcc = mfcc / (np.linalg.norm(mfcc, axis=1, keepdims=True) + 1e-6)
    return torch.from_numpy(mfcc)


def extract_mfccs(segment_root, save_to, config):
    videos, segments = [], []
    videos_sc = []

    for video_class in os.listdir(segment_root):
        for video in os.listdir(os.path.join(segment_root, video_class)):
            videos.append(os.path.join(video_class, video))
            segment_list = os.listdir(os.path.join(segment_root, video_class, video))
            for segment in segment_list:
                segments.append(os.path.join(segment_root, video_class, video, segment))
            videos_sc.append(len(segment_list))

    bar = tqdm(segments, ncols=150, unit='segment')
    features = []

    vidx = 0
    for idx, segment in enumerate(bar):
        feat = create_3_mfccs(segment, config)
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


def plot_waveform(segment):
    plt.figure(figsize=(10, 4))
    y, sr = lb.load(segment)
    lbd.waveshow(y, sr=sr)
    plt.title('Sound Wave')
    plt.tight_layout()
    plt.show()


def plot_mfcc(spec, win_length, config):
    plt.figure(figsize=(10, 4))
    lbd.specshow(spec, x_axis='time', y_axis='mel', sr=config.sampling_rate, win_length=win_length,
                 hop_length=config.hop_length, fmax=8000)
    plt.colorbar()
    plt.title(f'{win_length} MFCC')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Segment Audio Feature')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--hop_length', type=int, default=220)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--n_mfcc', type=int, default=20)
    parser.add_argument('--feature', type=str, default='MFCC', choices=['MFCC'])

    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.dataset_root, 'videos'))
    feature_dir = os.path.abspath(os.path.join(args.dataset_root, f'audio_features_{args.feature}_{args.duration}s'))

    video_cls = ['HighLight', 'Origin']
    videos = sorted([os.path.join(c, v) for c in video_cls for v in os.listdir(os.path.join(video_dir, c))])

    audio_segment_dir = os.path.abspath(os.path.join(args.dataset_root, f'audios_{args.duration}s'))

    segments = sorted([os.path.join(audio_segment_dir, v, segment) for v in videos for segment in
                       os.listdir(os.path.join(audio_segment_dir, v))])

    extract_mfccs(segment_root=audio_segment_dir, save_to=feature_dir, config=args)
