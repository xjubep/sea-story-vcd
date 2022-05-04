from collections import defaultdict
from abc import *
import numpy as np
import os
import json
import pandas as pd
import json

from VCD.datasets.mlvd import MLVD

# NSFW

'''
{nsfw_root}/
├── annotation/
├── videos/
├── frames/
└── meta/
'''


class NSFW(MLVD):
    def __init__(self, root):
        raise NotImplementedError
        super().__init__(root)
        self.videos, self.gt = self.parse_annotation()  # from annotation file
        self.videos, self.distract, self.meta, self.fc = self.read_metadata()  # from root directiory

    def parse_annotation(self):
        raise NotImplementedError

    @property
    def frame_annotation(self):
        """
        Scan parsed videos and filter unparsed videos, GT.
        Returns:
            videos : parsed core video
            distract : parsed distraction video
            metadata : metadata
            framecount : num of decoded frames for each videos
        """
        raise NotImplementedError
        frame_annotations = defaultdict(list)
        for video, ann in self.annotation.items():
            for gt in ann:
                sa, ea, b, sb, eb = gt
                if video != b and sa != sb and ea != eb:
                    cnt = min(ea - sa, eb - sb)
                    af = np.linspace(sa, ea, cnt, endpoint=False, dtype=np.int)
                    bf = np.linspace(sb, eb, cnt, endpoint=False, dtype=np.int)
                    frame_annotations[video] += [[f[0], b, f[1]] for f in zip(af, bf)]
                    frame_annotations[b] += [[f[1], video, f[0]] for f in zip(af, bf)]

    @property
    def video_annotation(self):
        raise NotImplementedError
