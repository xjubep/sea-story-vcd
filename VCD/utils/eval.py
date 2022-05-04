from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

import numpy as np
import faiss
import os


from VCD.utils import *


def vcdb_frame_retrieval(vcdb, features, index, chunk, margin, progress=False):
    avg_dist = AverageMeter()
    avg_rank = AverageMeter()
    avg_dist_per_query = AverageMeter()
    avg_rank_per_query = AverageMeter()

    annotations = defaultdict(set)
    for q_video, gt in vcdb.frame_annotation.items():
        q_video_idx = find_video_idx(q_video, vcdb.core_videos)
        for g in gt:
            r_video_idx = find_video_idx(g[0], vcdb.core_videos)
            for q, r in g[1]:
                annotations[index[q_video_idx][0] + q].add(index[r_video_idx][0] + r)

    query = list(annotations.keys())

    faiss.normalize_L2(features)
    IP_index = faiss.IndexFlatIP(features.shape[1])
    IP_index.add(features)

    results = dict()
    _iter = range(0, len(query), chunk)
    if progress:
        bar = tqdm(_iter, ncols=150)
    for i in _iter:
        q_feature_idx = query[i:i + chunk]
        dist, rank = IP_index.search(features[q_feature_idx], features.shape[0])

        for n, q in enumerate(q_feature_idx):
            for gt in annotations[q]:
                gt_rank = np.where(abs(rank[n] - gt) <= margin)[0][0]
                gt_dist = dist[n, gt_rank]
                results[(q, gt)] = (gt_dist, gt_rank)
                avg_rank.update(gt_rank)
                avg_dist.update(gt_dist)
        if progress:
            bar.set_description_str(f'Rank/Frame: {avg_rank.avg:4f}, Distance/Frame: {avg_dist.avg:4f}')
            bar.update()

    for q_video, gt in vcdb.frame_annotation.items():
        q_video_idx = find_video_idx(q_video, vcdb.core_videos)
        for g in gt:
            r_video_idx = find_video_idx(g[0], vcdb.core_videos)
            dist, rank = 0, 0
            for q, r in g[1]:
                q_feature_idx = index[q_video_idx][0] + q
                r_feature_idx = index[r_video_idx][0] + r
                dist += results[(q_feature_idx, r_feature_idx)][0]
                rank += results[(q_feature_idx, r_feature_idx)][1]

            avg_dist_per_query.update(dist / len(g[1]))
            avg_rank_per_query.update(rank / len(g[1]))
    if progress:
        bar.set_description_str(f'Rank/Frame: {avg_rank.avg:4f}, Rank/Seg :{avg_rank_per_query.avg:4f}, '
                                f'Distance/Frame: {avg_dist.avg:4f}, Distance/Seg: {avg_dist_per_query.avg:4f}')
        bar.close()
    return avg_rank.avg, avg_dist.avg, avg_rank_per_query.avg, avg_dist_per_query.avg


def vcdb_partial_copy_detection(vcdb, feature_root, param):
    paths = [os.path.join(feature_root, v + '.pth') for v in vcdb.all_videos]
    features, idx = load_feature(paths, progress=False)

    idx_table = {i: (n, i - l[0]) for n, l in enumerate(idx) for i in range(l[0], l[1])}
    func_mapping_idx = np.vectorize(lambda x: idx_table[x])

    index = faiss.IndexFlatIP(features.shape[1])
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)

    measure = {'tp_d': AverageMeter(),
               'tp_g': AverageMeter(),
               'all_d': AverageMeter(),
               'all_g': AverageMeter()}

    start_time = datetime.now()
    bar = tqdm(vcdb.annotation.items(), ncols=150, mininterval=1)
    for n, (query, gt) in enumerate(vcdb.annotation.items(), start=1):
        q_idx = np.where(vcdb.all_videos == query)[0][0]
        start, end = idx[q_idx]
        q_feat = features[start:end]
        D, I = index.search(q_feat, param[0])

        vidx, fidx = func_mapping_idx(I)

        tn = TN(D, vidx, fidx, *param[1:])
        candidate = tn.fit()

        # gt : ref_video idx, query_time, ref_time
        ground = [(np.where(vcdb.all_videos == g[2])[0][0], Period(g[0], g[1]), Period(g[3], g[4])) for g in gt]

        result = _match(candidate, ground)

        prec, prec_all, rec, rec_all = _update(measure, [*result, len(candidate), len(ground)])

        bar.set_description_str(f'{n:>3d} - ' \
                                f'fscore: {fscore(prec, rec):.4f} ({fscore(prec_all, rec_all):.4f}) ' \
                                f'precision: {prec:.4f} ({prec_all:.4f}) ' \
                                f'recall: {rec:.4f} ({rec_all:.4f})', refresh=False)
        bar.update()

    bar.close()

    return fscore(prec_all, rec_all), prec_all, rec_all, datetime.now() - start_time


# SP=|correctly retrieved segments|/|all retrieved segments|
# SR=|correctly retrieved segments|/|groundtruth copy segments|. I
def _match(path, gt):
    def vectorized_match(idx):
        y, x = divmod(idx, len(gt))
        if path[y][0] == gt[x][0] and path[y][1].is_overlap(gt[x][1]) and path[y][2].is_overlap(gt[x][2]):
            return 1  # x
        return 0

    d, g = 0, 0
    if len(path) and len(gt):
        correct = np.arange(0, len(path) * len(gt))
        ret = np.vectorize(vectorized_match)(correct).reshape(len(path), len(gt))
        d = np.count_nonzero(np.sum(ret, axis=1))
        g = np.count_nonzero(np.sum(ret, axis=0))

    return d, g


def _update(measure, values):
    measure['tp_d'].update(values[0])
    measure['tp_g'].update(values[1])
    measure['all_d'].update(values[2])
    measure['all_g'].update(values[3])

    prec = safe_ratio(measure['tp_d'].val, measure['all_d'].val)
    prec_all = safe_ratio(measure['tp_d'].sum, measure['all_d'].sum)
    rec = safe_ratio(measure['tp_g'].val, measure['all_g'].val)
    rec_all = safe_ratio(measure['tp_g'].sum, measure['all_g'].sum)

    return prec, prec_all, rec, rec_all
