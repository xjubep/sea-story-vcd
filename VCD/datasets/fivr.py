from collections import defaultdict
import numpy as np
import json
import os

# FIVR
# project : http://ndd.iti.gr/fivr/

# run below commands to create FIVR datasets root
# git clone https://github.com/MKLab-ITI/FIVR-200K
# mv FIVR-200K {FIVR root}
# pip install youtube_dl
# cd {FIVR root}
# python download_dataset.py -v ./videos -c {core}

# set FIVR root directory as below
'''
{FIVR_root}/
├── datasets/
│   ├── core.json
│   ├── annotation.json
│   ├── events.json
│   └── youtube_ids.txt
├── videos/
│   ├── xxxxxx.mp4
│   ├── ...
│   └── zzzzzz.mp4
├── frames/
└── meta/
'''

from VCD.datasets.mlvd import MLVD


class FIVR(MLVD):
    def __init__(self, root, positive=('ND', 'DS')):
        """
        'ND': Near-Duplicate - These are a special case of DSVs (all candidate scenes are duplicates with the query scenes).
        'DS': Duplicate Scene - DSVs are annotated with this label.
        'CS': Complementary Scene - CSVs are annotated with this label.
        'IS': Incident Scene - ISVs are annotated with this label.

        Task
        DSVR: ND,DS
        CSVR: ND,DS,CS
        ISVR: ND,DS,CS,IS
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

        annotation = json.load(open(f'{self.root}/dataset/annotation.json', 'r'))
        videos = set()  # core
        gt = defaultdict(list)
        for q, ann in annotation.items():
            qvid = os.path.join('core', q + '.mp4')
            videos.add(qvid)
            for label, ref in ann.items():
                r_vid = [os.path.join('core', r + '.mp4') for r in ref]
                videos.update(set(r_vid))
                if label in positive:
                    gt[qvid].extend(r_vid)

        return videos, dict(gt)


if __name__ == '__main__':
    dataset = FIVR(root='/mldisk/nfs_shared_/MLVD/FIVR', positive=('ND', 'DS', 'CS', 'IS'))
    print(dataset)
