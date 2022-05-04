import numpy as np
import argparse


def find_video_idx(name, videos):
    idx = np.where(videos == name)[0]
    video_idx = idx[0] if len(idx) != 0 else -1

    return video_idx


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
