import argparse
import os
import warnings

warnings.filterwarnings(action='ignore')

from tqdm import tqdm

import librosa as lb
import numpy as np

import joblib


def mono_to_color(x, eps=1e-6, mean=None, std=None):
    mean = mean or x.mean()
    std = std or x.std()
    x = (x - mean) / (std + eps)
    _min, _max = x.min(), x.max()

    if (_max - _min) > eps:
        v = np.clip(x, _min, _max)
        v = 255 * (v - _min) / (_max - _min)
        v = v.astype(np.uint8)
    else:
        v = np.zeros_like(x, dtype=np.uint8)

    return v


def crop_or_pad(y, length):
    y = np.concatenate([y, np.zeros(length - len(y))])
    n_repeats = length // len(y)
    epsilon = length % len(y)
    y = np.concatenate([y] * n_repeats + [y[:epsilon]])
    return y


class MelSpecComputer:
    def __init__(self, sr, n_mels, fmin, fmax, **kwargs):
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        kwargs['n_fft'] = kwargs.get('n_fft', self.sr // 10)
        kwargs['hop_length'] = kwargs.get('hop_length', self.sr // (10 * 4))
        self.kwargs = kwargs

    def __call__(self, y):
        melspec = lb.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax,
            **self.kwargs, )
        melspec = lb.power_to_db(melspec).astype(np.float32)
        return melspec


class AudioToImage:
    def __init__(self, sr=22050, n_mels=128, fmin=0, fmax=None, duration=10, step=None, res_type="kaiser_fast",
                 resample=True):

        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or self.sr // 2

        self.duration = duration
        self.audio_length = self.duration * self.sr
        self.step = step or self.audio_length

        self.res_type = res_type
        self.resample = resample

        self.mel_spec_computer = MelSpecComputer(sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)

    def audio_to_image(self, audio):
        melspec = self.mel_spec_computer(audio)
        image = mono_to_color(melspec)
        return image

    def __call__(self, root_dir, save_dir, video, save=True):
        audio, orig_sr = lb.load(os.path.join(root_dir, 'audios', f'{video}.aac'))

        if self.resample and orig_sr != self.sr:
            audio = lb.resample(audio, orig_sr, self.sr, res_type=self.res_type)

        audios = [audio[i:i + self.audio_length] for i in range(0, max(1, len(audio)), self.step)]
        if len(audio) % self.audio_length != 0:  # need padding
            audios[-1] = crop_or_pad(audios[-1], length=self.audio_length)

        images = [self.audio_to_image(audio) for audio in audios]
        images = np.stack(images)

        if save:
            save_path = os.path.join(save_dir, video)
            np.save(f'{save_path}.npy', images)
        else:
            return os.path.join(root_dir, 'audios', f'{video}.aac'), images


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Segment Mel-Spectrogram')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--duration', type=int, default=15)
    parser.add_argument('--sampling_rate', type=int, default=22050)
    parser.add_argument('--worker', type=int, default=8)
    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.dataset_root, 'videos'))
    mels_dir = os.path.abspath(os.path.join(args.dataset_root, f'mels_{args.duration}s'))

    video_cls = ['HighLight', 'Origin']
    videos = sorted([os.path.join(c, v) for c in video_cls for v in os.listdir(os.path.join(video_dir, c))])

    if os.path.exists(mels_dir):
        print(f'Mel-Spectrogram directory {mels_dir} is already exist')
        exit(1)

    for c in video_cls:
        os.makedirs(os.path.join(mels_dir, c))

    pool = joblib.Parallel(args.worker)
    converter = AudioToImage(duration=args.duration, sr=args.sampling_rate,
                             step=int(args.duration * args.sampling_rate))
    mapper = joblib.delayed(converter)
    tasks = [mapper(args.dataset_root, mels_dir, v) for v in videos]

    pool(tqdm(tasks))
