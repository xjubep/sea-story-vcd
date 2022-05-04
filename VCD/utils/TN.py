from collections import defaultdict
from typing import Union

import numpy as np


class Period(object):
    # half-closed form [a, b)
    def __init__(self, s, e):
        self.start, self.end = (s, e) if s < e else (e, s)

    def __repr__(self):
        return '{} - {}'.format(self.start, self.end)

    @property
    def length(self):
        return self.end - self.start

    def __add__(self, v: Union[int, float]):
        self.start += v
        self.end += v
        return self

    def __sub__(self, v: Union[int, float]):
        self.start -= v
        self.end -= v
        return self

    def __mul__(self, v: Union[int, float]):
        self.start *= v
        self.end *= v
        return self

    def is_overlap(self, o):
        assert isinstance(o, Period)
        return not ((self.end <= o.start) or (o.end <= self.start))

    def is_in(self, o):
        assert isinstance(o, Period)
        return o.start <= self.start and self.end <= o.end

    # self.start <= o.start <= o.end <= self.end
    def is_wrap(self, o):
        assert isinstance(o, Period)
        return self.start <= o.start and o.end <= self.end

    def intersection(self, o):
        assert isinstance(o, Period)
        return Period(max(self.start, o.start), min(self.end, o.end)) if self.is_overlap(o) else None

    # if not overlap -> self
    def union(self, o):
        assert isinstance(o, Period)
        return Period(min(self.start, o.start), max(self.end, o.end)) if self.is_overlap(o) else None

    def IOU(self, o):
        try:
            intersect = self.intersection(o)
            union = self.union(o)
            iou = intersect.length / union.length
        except:
            iou = 0
        return iou


class TN(object):
    def __init__(self, D, video_idx, frame_idx, FEAT_INTV=1, TEMP_WND=10, PATH_THR=5, SCORE_THR=-1):
        self.FEAT_INTV = FEAT_INTV
        self.TEMP_WND = TEMP_WND
        self.PATH_THR = PATH_THR
        self.SCORE_THR = SCORE_THR

        # [# of query index, topk]
        self.video_index = video_idx
        self.frame_index = frame_idx
        self.dist = D

        self.query_length = D.shape[0]
        self.topk = D.shape[1]

        # dist, count, query start, reference start
        self.paths = np.empty((*D.shape, 4), dtype=object)

    def find_previous_linkable_nodes(self, t, r):
        video_idx, frame_idx = self.video_index[t, r], self.frame_index[t, r]
        min_prev_time = max(0, t - self.TEMP_WND)

        # find previous nodes that have (same video index) and (frame timestamp - wnd <= previous frame timestamp < frame timestamp)
        time, rank = np.where((self.dist[min_prev_time:t, ] >= self.SCORE_THR) &
                              (self.video_index[min_prev_time:t, ] == video_idx) &
                              (self.frame_index[min_prev_time:t, ] >= frame_idx - self.TEMP_WND) &
                              (self.frame_index[min_prev_time:t, ] < frame_idx)
                              )

        return np.stack((time + min_prev_time, rank), axis=-1)

    def fit(self):
        # find linkable nodes
        for time in range(self.query_length):
            for rank in range(self.topk):
                prev_linkable_nodes = self.find_previous_linkable_nodes(time, rank)

                if not len(prev_linkable_nodes):
                    self.paths[time, rank] = [self.dist[time, rank],
                                              1,
                                              time,
                                              self.frame_index[time, rank]]
                else:
                    # priority : count, path length, path score
                    prev_time, prev_rank = max(prev_linkable_nodes, key=lambda x: (self.paths[x[0], x[1], 1],
                                                                                   self.frame_index[time, rank] -
                                                                                   self.paths[x[0], x[1], 3],
                                                                                   self.paths[x[0], x[1], 0]
                                                                                   ))
                    prev_path = self.paths[prev_time, prev_rank]
                    self.paths[time, rank] = [prev_path[0] + self.dist[time, rank],
                                              prev_path[1] + 1,
                                              prev_path[2],
                                              prev_path[3]]

        # connect and filtering paths
        candidate = defaultdict(list)
        for time in reversed(range(self.query_length)):
            for rank in range(self.topk):
                score, count, q_start, r_start = self.paths[time, rank]
                if count >= self.PATH_THR:
                    video_idx, frame_idx = self.video_index[time, rank], self.frame_index[time, rank]
                    q = Period(q_start, time + 1)
                    r = Period(r_start, frame_idx + 1)
                    path = (video_idx, q, r, score, count)
                    # remove include path
                    flag = True
                    for n, c in enumerate(candidate[video_idx]):
                        if path[1].is_wrap(c[1]) and path[2].is_wrap(c[2]):
                            candidate[video_idx][n] = path
                            flag = False
                            break
                        elif path[1].is_in(c[1]) and path[2].is_in(c[2]):
                            flag = False
                            break
                    if flag:
                        candidate[video_idx].append(path)

        # remove overlap path
        for video, path in candidate.items():
            candidate[video] = self.nms_path(path)

        candidate = [[c[0], c[1] * self.FEAT_INTV, c[2] * self.FEAT_INTV, c[3], c[4]]
                     for cc in candidate.values() for c in cc]
        return candidate

    def nms_path(self, path):
        l = len(path)
        path = np.array(sorted(path, key=lambda x: (x[4], x[3], x[2].length, x[1].length), reverse=True))

        keep = np.array([True] * l)
        overlap = np.vectorize(lambda x, a: x.is_overlap(a))
        for i in range(l - 1):
            if keep[i]:
                keep[i + 1:] = keep[i + 1:] & \
                               (~(overlap(path[i + 1:, 1], path[i, 1]) & overlap(path[i + 1:, 2], path[i, 2])))
        path = path.tolist()
        path = [path[n] for n in range(l) if keep[n]]

        return path
