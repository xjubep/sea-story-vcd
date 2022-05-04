from .fivr import FIVR
from .vcdb import VCDB
from .cc_web import CC_WEB
from .nsfw import NSFW
from .dataset import ListDataset, TripletDataset

DATASETS = ['VCDB', 'CC_WEB', 'FIVR', 'NSFW']


def get_dataset(name, root, **kwargs):
    assert name in DATASETS
    if name == 'VCDB':
        module = VCDB
    elif name == 'CC_WEB':
        module = CC_WEB
    elif name == 'FIVR':
        module = FIVR
    elif name == 'NSFW':
        module = NSFW

    return module(root)
