import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from VCD.utils import *
from VCD.utils.eval import _match, _update


def parse(ann):
    a, b, sa, ea, sb, eb = ann
    times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in [sa, ea, sb, eb]]
    return [a, b, *times]


def read_annotations(sea_root):
    annotations = defaultdict(list)
    for r, d, f in os.walk(os.path.join(sea_root, 'annotations')):
        for v in f:
            annotation = pd.read_csv(os.path.join(r, v))
            for idx in range(annotation.shape[0]):
                a, b, sa, ea, sb, eb = parse(annotation.loc[idx])
                video_a = os.path.join('HighLight', a)
                video_b = os.path.join('Origin', b)
                annotations[video_a] += [[sa, ea, video_b, sb, eb]]
    return annotations


def partial_copy_detection(sea_root, feature_root, annotations, param):
    sea_highlight_videos = np.array([os.path.join('HighLight', v) for v in
                                     os.listdir(os.path.join(sea_root, 'videos', 'HighLight'))])
    sea_origin_videos = np.array(
        [os.path.join('Origin', v) for v in os.listdir(os.path.join(sea_root, 'videos', 'Origin'))])

    highlight_paths = [os.path.join(feature_root, v + '.pth') for v in sea_highlight_videos]
    origin_paths = [os.path.join(feature_root, v + '.pth') for v in sea_origin_videos]

    hi_features, hi_idx = load_feature(highlight_paths, progress=False)
    ori_features, ori_idx = load_feature(origin_paths, progress=False)

    ori_idx_table = {i: (n, i - l[0]) for n, l in enumerate(ori_idx) for i in range(l[0], l[1])}

    func_mapping_ori_idx = np.vectorize(lambda x: ori_idx_table[x])

    faiss.normalize_L2(ori_features)

    ori_index = faiss.IndexFlatIP(ori_features.shape[1])
    ori_index = faiss.index_cpu_to_all_gpus(ori_index)
    ori_index.add(ori_features)

    measure = {'tp_d': AverageMeter(),
               'tp_g': AverageMeter(),
               'all_d': AverageMeter(),
               'all_g': AverageMeter()}

    start_time = datetime.now()
    bar = tqdm(annotations.items(), ncols=150, mininterval=1)
    for n, (query, gt) in enumerate(annotations.items(), start=1):
        q_idx = np.where(sea_highlight_videos == query)[0][0]
        start, end = hi_idx[q_idx]
        q_feat = hi_features[start:end]
        D, I = ori_index.search(q_feat, param[0])

        vidx, fidx = func_mapping_ori_idx(I)

        tn = TN(D, vidx, fidx, *param[1:])
        candidate = tn.fit()

        # gt : ref_video idx, query_time, ref_time
        ground = [(np.where(sea_origin_videos == g[2])[0][0], Period(g[0], g[1]), Period(g[3], g[4])) for g in gt]
        result = _match(candidate, ground)

        prec, prec_all, rec, rec_all = _update(measure, [*result, len(candidate), len(ground)])

        bar.set_description_str(f'{n:>3d} - '
                                f'fscore: {fscore(prec, rec):.4f} ({fscore(prec_all, rec_all):.4f}) '
                                f'precision: {prec:.4f} ({prec_all:.4f}) '
                                f'recall: {rec:.4f} ({rec_all:.4f})', refresh=False)
        bar.update()

    bar.close()

    return fscore(prec_all, rec_all), prec_all, rec_all, datetime.now() - start_time


def performance(sea_root, feature_root, csv_path, feature_model, duration):
    f = open(f"{csv_path}/{feature_model}.csv", "a", newline='')
    w = csv.writer(f)
    w.writerow(['topk', 'window', 'path_thr', 'score_thr', 'fscore', 'precision', 'recall', 'time'])

    annotations = read_annotations(sea_root)

    feature_intv = duration
    topks = [25, 50, 75]
    windows = [1, 2, 3, 4]
    path_thrs = [1, 2, 3, 4, 5]
    score_thrs = [-1, 0.8, 0.7, 0.5, 0.3]

    for topk in topks:
        for window in windows:
            for path_thr in path_thrs:
                for score_thr in score_thrs:
                    param = [topk, feature_intv, window, path_thr, score_thr]
                    print(param)
                    fscore, prec, rec, time = partial_copy_detection(sea_root, feature_root, annotations, param)
                    w.writerow([topk, window, path_thr, score_thr, fscore, prec, rec, time])

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temporal network with random parameter')
    parser.add_argument('--dataset_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--duration', type=int, default=5)
    parser.add_argument('--mode', type=str, default='concat', choices=['sum', 'concat'])
    parser.add_argument('--feature', type=str, default='MFCC', choices=['MFCC'])

    args = parser.parse_args()
    print(args)

    feature_model = f'multi_features_{args.feature}_{args.duration}s_{args.mode}'

    feature_path = os.path.abspath(os.path.join(args.dataset_root, f'{feature_model}'))

    csv_path = os.path.abspath(os.path.join(args.dataset_root, 'results', 'temporal_csv'))
    if not os.path.isdir(csv_path):
        os.makedirs(csv_path, exist_ok=True)

    performance(args.dataset_root, feature_path, csv_path, feature_model, args.duration)
