from collections import defaultdict
import pandas as pd
import numpy as np
import json
import os

from VCD.datasets.mlvd import MLVD

# CC_WEB_VIDEO Dataset
# download at http://vireo.cs.cityu.edu.hk/webvideo/Download.htm
# set CC_WEB root directory as below
'''
{root}/
├── GT/
│   ├── GT1.rst
│   ├── ...
│   └── GT24rst
├── videos/
│   ├── 1_1_Y.flv
│   ├── ...
│   └── 24_xxx.flv
├── frames/
├── meta/
├── Video_List.txt
└── Seed.txt
'''


class CC_WEB(MLVD):
    def __init__(self, root, positive='ESVML'):
        """
        positive
        'E': Exactly duplicate,
        'S': Similar,
        'V': Different version,
        'M': Major change,
        'L': Long version,
        'X': Dissimilar video,
        '-1': Video does not exist
        """
        super().__init__(root)
        self._videos, self._gt = self.parse_annotation(positive)
        self._videos, self._distract, self._meta, self._fc = self.read_metadata()

    def parse_annotation(self, positive):
        """
        Parse annotation file
        Returns:
            videos : core video
            gt : positive videos per each query video
        """

        videos = pd.read_csv(os.path.join(self.root, 'Video_List.txt'), sep='\t', header=None)[3].to_numpy()
        query_idx = pd.read_csv(os.path.join(self.root, 'Seed.txt'), sep='\t', header=None)[1].to_numpy()
        gt = defaultdict(list)
        for n, q in enumerate(query_idx, start=1):
            for l in open(os.path.join(self.root, 'GT', f'GT_{n}.rst')).readlines():
                idx, label = l.strip().split()
                if label in positive:
                    gt[videos[q - 1]].append(videos[int(idx) - 1])

        return videos, dict(gt)


if __name__ == '__main__':
    # datasets = CC_WEB(root='/hdd/ms/VCD-baseline/datasets/CC_WEB', positive='E')
    dataset = CC_WEB(root='/mldisk/nfs_shared_/MLVD/CC_WEB', positive='E')
    print(dataset)
