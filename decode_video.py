import argparse
import os
import tqdm
from VCD.utils import *
from datetime import datetime

logger = None


def extract_frame(video, frame_dir, decode_rate):
    cmd = f'ffmpeg -ss 00:00:0 -i "{video}" -r {decode_rate} -f image2 "{frame_dir}/%06d.jpg"'
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract frames')
    parser.add_argument('--root', type=str, help='root directory')
    parser.add_argument('--video_dir', type=str, default='videos')
    parser.add_argument('--frame_dir', type=str, default='frames')
    parser.add_argument('--fps', type=int, default=1)

    args = parser.parse_args()
    args.root = '/mldisk/nfs_shared_/sy/sea_story'

    video_dir = os.path.abspath(os.path.join(args.root, args.video_dir))
    frame_dir = os.path.abspath(os.path.join(args.root, args.frame_dir))

    log = os.path.join(args.root, f'extract_{args.frame_dir}.log')
    if os.path.exists(log):
        log = os.path.join(args.root, f'extract_{args.frame_dir}_{datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")}.log')

    logger = initialize_log(log)

    if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) != 0:
        logger.info(f'Frame directory {frame_dir} is already exist')
        exit(1)

    videos = sorted(
        [os.path.relpath(os.path.join(r, v), video_dir) for r, d, f in os.walk(video_dir, followlinks=True) for v in f
         if v.split('.')[1].lower() in VIDEO_EXTENSION])

    logger.info(f'Decode {len(videos)} videos, {args.fps} fps')
    logger.info(f'Videos ... {video_dir}')
    logger.info(f'Frames ... {frame_dir}')
    logger.info(f'log ... {os.path.abspath(log)}')

    bar = tqdm.tqdm(videos, mininterval=1, ncols=150)

    for v in bar:
        os.makedirs(os.path.join(frame_dir, v))
        extract_frame(os.path.join(video_dir, v), os.path.join(frame_dir, v), args.fps)
