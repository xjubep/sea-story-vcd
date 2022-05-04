from collections import defaultdict
import numpy as np
import os

from VCD.datasets.mlvd import MLVD

# VCDB
# download at http://www.yugangjiang.info/research/VCDB/index.html
# set VCDB root directory as below
'''
{VCDB_root}/
├── annotation/
│   ├── baggio_penalty_1994.txt
│   ├── ...
│   └── zidane_headbutt.txt
├── videos/
│   ├── core_dataset/
│   │   ├── baggio_penalty_1994
│   │   │   ├── 5c5714c0a56fd2a96f99db2f59b0d03659d77cdf.flv
│   │   │   ├── ...
│   │   │   └── e901a631b00f4ad0c9d161d686fac1339e1e3535.flv
│   │   ├── ...
│   │   └── zidane_headbutt
│   │       ├── 0ab11b52561e9255423b01f29f7904a6dcadd87b.flv
│   │       ├── ...
│   │       └── e2d1ca4b2657cd82092765801adda59a7f71bc56.mp4
│   └── distract_dataset/
├── frames/
└── meta/
'''


class VCDB(MLVD):
    def __init__(self, root):

        super().__init__(root)
        self._videos, self._gt = self.parse_annotation()  # from annotation file
        self._videos, self._distract, self._meta, self._fc = self.read_metadata()  # from root directiory

    def parse_annotation(self):
        """
        Parse annotation file
        Returns:
            videos : core video
            gt : positive videos per each query video
        """

        def parse(ann):
            a, b, *times = ann.strip().split(',')
            times = [sum([60 ** (2 - n) * int(u) for n, u in enumerate(t.split(':'))]) for t in times]
            return [a, b, *times]

        videos = set()
        annotations = defaultdict(list)
        for r, d, f in os.walk(os.path.join(self.root, 'annotation')):
            for g in f:
                for l in open(os.path.join(r, g)).readlines():
                    group = os.path.splitext(g)[0]
                    a, b, sa, ea, sb, eb = parse(l)
                    video_a = os.path.join('core_dataset', group, a)
                    video_b = os.path.join('core_dataset', group, b)
                    videos.add(video_a)
                    annotations[video_a] += [[sa, ea, video_b, sb, eb]]
                    if a != b:
                        videos.add(video_b)
                        annotations[video_b] += [[sb, eb, video_a, sa, ea]]

        return videos, annotations

    @property
    def frame_annotation(self):
        frame_annotations = defaultdict(list)
        for q, ann in self.annotation.items():
            for gt in ann:
                sa, ea, b, sb, eb = gt
                if q != b and sa != sb and ea != eb:
                    cnt = min(ea - sa, eb - sb)
                    af = np.linspace(sa, ea, cnt, endpoint=False, dtype=np.int)
                    bf = np.linspace(sb, eb, cnt, endpoint=False, dtype=np.int)
                    frame_annotations[q].append((b, [(f[0], f[1]) for f in zip(af, bf)]))

        return frame_annotations

    @property
    def video_annotation(self):
        raise NotImplementedError


if __name__ == '__main__':
    dataset = VCDB(root='/mldisk/nfs_shared_/MLVD/VCDB-core')
    print(dataset)
    # print(dataset.frame_annotation)
    dataset = VCDB(root='/mldisk/nfs_shared_/MLVD/VCDB')
    print(dataset)

    # print(dataset.annotation)

    print(dataset.frame_annotation['core_dataset/zidane_headbutt/0dbcff0b6d671e1579ad468eed54d84f7e9b7289.mp4'])
    print(dataset.frame_annotation.keys().__len__())