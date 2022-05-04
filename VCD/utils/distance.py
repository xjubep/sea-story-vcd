import numpy as np
import faiss


def l2_distance(a, b):
    idx = faiss.IndexFlatL2(a.shape[1])
    idx.add(a)
    dist, _ = idx.search(b, 1)
    return dist[0][0]

def cos_distance(a, b):
    idx = faiss.IndexFlatIP(a.shape[1])
    idx.add(a)
    dist, _ = idx.search(b, 1)
    return dist[0][0]