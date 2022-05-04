from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
import torch
import os

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

def load_file(p):
    _, ext = os.path.splitext(p)
    if ext in ['.pt', '.pth']:
        f = torch.load(p).numpy()
    elif ext in ['.npy']:
        f = np.load(p)
    else:
        raise TypeError(f'feature extension {ext} isn\'t supported')

    return f, f.shape[0]


def load_feature(paths, process=4, progress=False, desc='Load features'):
    def update():
        if progress:
            bar.update()

    bar = tqdm(paths, desc=desc, ncols=150, mininterval=1) if progress else None

    pool = Pool(process)
    results = [pool.apply_async(load_file, args=[p], callback=lambda *a: update()) for p in
               paths]
    pool.close()
    pool.join()

    if progress:
        bar.close()

    results = [(f.get()) for f in results]
    features = np.concatenate([r[0] for r in results])
    length = [r[1] for r in results]

    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    return features, index


@torch.no_grad()
def extract_feature(model, loader, args, progress=True, **kwargs):
    model.eval()
    features = []
    bar = tqdm(loader, desc='Extract features', ncols=150, unit='batch') if progress else loader
    for i, (path, frame) in enumerate(bar):
        if args.amp:
            with autocast():
                out = model(frame, **kwargs).cpu()
                features.append(out)
        else:
            out = model(frame, **kwargs).cpu()
            features.append(out)

    features = torch.cat(features).numpy()
    return features
