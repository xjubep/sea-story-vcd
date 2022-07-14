import argparse
import os
import subprocess
from datetime import timedelta

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from VCD.utils.TN import TN
from VCD.utils.load import load_feature


def Period_to_str(Period_sec):
    return str(timedelta(seconds=int(Period_sec)))


def time_to_Fmt(start, end):
    return start + ' - ' + end


def ffmpeg_subprocess(cmd):
    p = subprocess.Popen(args=cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False)
    _ = p.communicate()


def vcdb_partial_copy_detection(sea_root, feature_root, result_path, param):
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

    # faiss.normalize_L2(hi_features)
    faiss.normalize_L2(ori_features)

    ori_index = faiss.IndexFlatIP(ori_features.shape[1])
    ori_index = faiss.index_cpu_to_all_gpus(ori_index)
    ori_index.add(ori_features)

    save_path = os.path.join(result_path,
                             f'tk{param[0]:03d}_in{param[1]:02d}_wn{param[2]:02d}_pt{param[3]:01d}_sc{param[4]:0.1f}')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    bar = tqdm(sea_highlight_videos, ncols=120, mininterval=1, position=0)
    for n, (query) in enumerate(sea_highlight_videos, start=1):
        q_idx = np.where(sea_highlight_videos == query)[0][0]

        start, end = hi_idx[q_idx]
        q_feat = hi_features[start:end]
        faiss.normalize_L2(q_feat)
        D, I = ori_index.search(q_feat, param[0])

        vidx, fidx = func_mapping_ori_idx(I)

        tn = TN(D, vidx, fidx, *param[1:])
        candidate = tn.fit()
        candidate = sorted(candidate, key=lambda x: -x[3])

        # topk, feat interval, window, path threshold, score threshold
        save_query_path = os.path.join(save_path, os.path.basename(query))
        result_csv = f'{save_query_path}.csv'
        if not os.path.isdir(save_query_path):
            os.makedirs(save_query_path)

        result = []
        query_bar = tqdm(candidate, ncols=120, mininterval=1, position=1)
        for i, (r_idx, q_t, r_t, sc, feat_num) in enumerate(candidate):
            q_t_start, q_t_end = Period_to_str(q_t.start), Period_to_str(q_t.end)
            r_t_start, r_t_end = Period_to_str(r_t.start), Period_to_str(r_t.end)
            q_t, r_t = time_to_Fmt(q_t_start, q_t_end), time_to_Fmt(r_t_start, r_t_end)
            result.append({'query': sea_highlight_videos[q_idx], 'ref': sea_origin_videos[r_idx],
                           'query_time': q_t, 'ref_time': r_t, 'score': sc, 'feat_num': feat_num})

            query_v = os.path.join('/mldisk/nfs_shared_/sy/sea_story/videos', sea_highlight_videos[q_idx])
            ref_v = os.path.join('/mldisk/nfs_shared_/sy/sea_story/videos', sea_origin_videos[r_idx])

            cmd1 = ['ffmpeg',
                    '-i', f'{query_v}', '-ss', q_t_start, '-to', q_t_end,
                    '-vcodec', 'copy', '-acodec', 'copy',
                    f'{save_query_path}/{i + 1:03d}_query.mp4']

            cmd2 = ['ffmpeg',
                    '-i', f'{ref_v}', '-ss', r_t_start, '-to', r_t_end,
                    '-vcodec', 'copy', '-acodec', 'copy',
                    f'{save_query_path}/{i + 1:03d}_ref.mp4']

            cmd3 = ['ffmpeg',
                    '-i', f'{save_query_path}/{i + 1:03d}_query.mp4',
                    '-vf', "select=\'eq(pict_type,I)\', scale=320:180, tile=6x6",
                    '-frames:v', '1',
                    f'{save_query_path}/{i + 1:03d}_query.jpg']

            cmd4 = ['ffmpeg',
                    '-i', f'{save_query_path}/{i + 1:03d}_ref.mp4',
                    '-vf', "select=\'eq(pict_type,I)\', scale=320:180, tile=6x6",
                    '-frames:v', '1',
                    f'{save_query_path}/{i + 1:03d}_ref.jpg']

            ffmpeg_subprocess(cmd1)
            ffmpeg_subprocess(cmd2)
            ffmpeg_subprocess(cmd3)
            ffmpeg_subprocess(cmd4)
            query_bar.update()

        query_bar.close()
        result = pd.DataFrame(result)
        result.to_csv(result_csv, index=False)

        bar.update()

    bar.close()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Partial copy detection for Sea Story.")
    parser.add_argument('--sea_root', type=str, default='/mldisk/nfs_shared_/sy/sea_story')
    parser.add_argument('--feature_path', type=str,
                        default='/mldisk/nfs_shared_/sy/sea_story/multi_features_MFCC_5s_concat')
    parser.add_argument('--result_path', type=str, default='/mldisk/nfs_shared_/sy/sea_story/results/MFCC')

    # TN - parameters
    parser.add_argument('--topk', type=int, default=25)
    parser.add_argument('--feature_intv', type=int, default=5)
    parser.add_argument('--window', type=int, default=3)
    parser.add_argument('--path_thr', type=int, default=5)
    parser.add_argument('--score_thr', type=float, default=0.8)

    args = parser.parse_args()
    print(args)

    param = [args.topk, args.feature_intv, args.window, args.path_thr, args.score_thr]

    vcdb_partial_copy_detection(args.sea_root, args.feature_path, args.result_path, param)
