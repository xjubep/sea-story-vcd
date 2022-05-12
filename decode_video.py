import argparse
import os
import subprocess
from datetime import datetime

import tqdm

from VCD.utils import *

logger = None


def ffmpeg_subprocess(cmd):
    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    _ = p.communicate()


def extract_frame(video, frame_dir, decode_rate):
    cmd = ['ffmpeg',
           '-i', f'{video}', '-ss', '00:00:0',
           '-r', f'{decode_rate}', '-f', 'image2', f'{frame_dir}/%06d.jpg']
    ffmpeg_subprocess(cmd)


def extract_audio(video, audio_dir):
    cmd = ['ffmpeg',
           '-i', f'{video}', '-vn', '-acodec', 'copy',
           f'{audio_dir}.aac']
    ffmpeg_subprocess(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames and audios')
    parser.add_argument('--root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--audio_dir', type=str, default='audios')
    parser.add_argument('--frame_dir', type=str, default='frames')
    parser.add_argument('--fps', type=int, default=1)

    args = parser.parse_args()

    video_dir = os.path.abspath(os.path.join(args.root, args.video_dir))
    audio_dir = os.path.abspath(os.path.join(args.root, args.audio_dir))
    frame_dir = os.path.abspath(os.path.join(args.root, args.frame_dir))

    log = os.path.join(args.root, f'extract_{args.frame_dir}.log')
    if os.path.exists(log):
        log = os.path.join(args.root, f'extract_{args.frame_dir}_{datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")}.log')

    logger = initialize_log(log)

    if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) != 0:
        logger.info(f'Frame directory {frame_dir} is already exist')
        exit(1)

    if os.path.exists(audio_dir) and len(os.listdir(audio_dir)) != 0:
        logger.info(f'Audio directory {audio_dir} is already exist')
        exit(1)

    video_cls = ['HighLight', 'Origin']
    videos = sorted([os.path.join(c, v) for c in video_cls for v in os.listdir(os.path.join(video_dir, c))])

    logger.info(f'Decode {len(videos)} videos, {args.fps} fps')
    logger.info(f'Videos ... {video_dir}')
    logger.info(f'Frames ... {frame_dir}')
    logger.info(f'Audios ... {audio_dir}')
    logger.info(f'log ... {os.path.abspath(log)}')

    bar = tqdm.tqdm(videos, mininterval=1, ncols=150)

    for c in video_cls:
        os.makedirs(os.path.join(audio_dir, c))

    for v in bar:
        os.makedirs(os.path.join(frame_dir, v))
        extract_frame(os.path.join(video_dir, v), os.path.join(frame_dir, v), args.fps)
        extract_audio(os.path.join(video_dir, v), os.path.join(audio_dir, v))
