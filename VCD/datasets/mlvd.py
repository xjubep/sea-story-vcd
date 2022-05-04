from abc import *
import numpy as np
import json
import os


# Multimedia Lab Video Dataset
class MLVD(metaclass=ABCMeta):
    def __init__(self, root):
        self.root = root
        self._videos = None
        self._distract = None
        self._gt = None
        self._meta = None
        self._fc = None

    def __repr__(self):
        msg = '\n'.join([f'{self.__class__.__name__}',
                         f'  Videos : {len(self.all_videos)}',
                         f'  Core Videos: {len(self.core_videos)}',
                         f'  Distraction Videos: {len(self.distract_videos)}',
                         f'  Query: {len(self.annotation.keys())}',
                         ])

        return msg

    @abstractmethod
    def parse_annotation(self, *args, **kwargs):
        pass

    @property
    def frame_root(self):
        return os.path.join(self.root, 'frames')

    @property
    def video_root(self):
        return os.path.join(self.root, 'videos')

    @property
    def core_videos(self):
        assert self._videos is not None
        return self._videos

    @property
    def distract_videos(self):
        assert self._distract is not None
        return self._distract

    @property
    def all_videos(self):
        assert self._videos is not None
        assert self._distract is not None
        return np.array(list(self._videos) + list(self._distract))

    @property
    def annotation(self):
        return self._gt

    def read_metadata(self):
        """
        Scan parsed videos and filter unparsed videos, GT.
        Returns:
            videos : parsed core video
            distract : parsed distraction video
            metadata : metadata
            framecount : num of decoded frames for each videos
        """
        assert self._videos is not None
        assert self._gt is not None

        frame_count = json.load(open(os.path.join(self.root, 'meta', 'frames.json')))
        metadata = json.load(open(os.path.join(self.root, 'meta', 'metadata.json')))

        annotation_video = set(self._videos)
        parsed_video = set(metadata.keys())

        core = annotation_video.intersection(parsed_video)
        miss = np.array(list(annotation_video - parsed_video))
        if len(miss) != 0:
            print(f'{len(miss)} core videos are missed.')
            self._gt = {q: [g for g in gt if g in core] for q, gt in self._gt.items() if q in core}

        distract = sorted(list(parsed_video - core))
        core = sorted(list(core))

        return np.array(core), np.array(distract), metadata, frame_count

    def get_framecount(self, video):
        assert self._fc is not None
        return self._fc[video]

    def get_frames(self, video):
        assert self._fc is not None
        frames = np.char.zfill(np.arange(1, self.get_framecount(video) + 1).astype(np.str), 6)
        return np.vectorize(lambda x: os.path.join(self.frame_root, video, x + '.jpg'))(frames)
