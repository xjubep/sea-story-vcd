import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from VCD.utils import *

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'NanumGothic'

sea_root = '/mldisk/nfs_shared_/sy/sea_story'
feat_dir = os.path.abspath(os.path.join(sea_root, f'multi_features_MFCC_5s_concat'))

video_cls = ['HighLight', 'Origin']

sea_all_videos = np.array(
    [os.path.join(cls, v) for cls in video_cls for v in os.listdir(os.path.join(sea_root, 'videos', cls))])
all_paths = [os.path.join(feat_dir, v + '.pth') for v in sea_all_videos]
features, idx = load_feature(all_paths, progress=False)

data = []
videos = []
for video in sea_all_videos:
    video_feat = torch.load(os.path.join(feat_dir, video + '.pth'))
    vs = video.split('/')
    videos.append(vs[0][0] + vs[1][0])
    for seg_feat in video_feat:
        data.append({'feat': seg_feat, 'mean': seg_feat.mean(), 'std': seg_feat.std(), 'video': vs[0][0] + vs[1][0]})
data = pd.DataFrame(data)
data = data.astype({'mean': 'float', 'std': 'float'})

pca = PCA(n_components=2)
pca.fit(features)
features_reduced = pca.transform(features)

pca_df = pd.DataFrame(data=features_reduced
                      , columns=['principal component 1', 'principal component 2'])
pca_df['video'] = data['video']

# 매치 정보는 highlight_origin.xlsx 참고
colors = {
    'H좋': '#B71C1C', 'O치': '#E57373', 'H막': '#4A148C', 'O카': '#BA68C8', 'H그': '#1A237E', 'O마': '#7986CB',
    'H개': '#01579B', 'O초': '#4FC3F7', 'H달': '#004D40', 'O극': '#4DB6AC', 'H무': '#33691E', 'O야': '#AED581',
    'H세': '#F57F17', 'O세': '#FFD54F', 'H아': '#BF360C', 'O오': '#FF8A65', 'H코': '#212121', 'O코': '#E0E0E0',
    'H야': '#880E4F', 'O트': '#F06292',
}

for k, v in colors.items():
    if k == 'H좋' or k == 'O치':
        pass
    else:
        colors[k] = '#FFFFFF'

markers = {}
for v in videos:
    markers[v] = '^' if v[0] == 'H' else '.'

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('Multi Feature PCA (MFCC concat)', fontsize=20)

for video in videos:
    indicesToKeep = pca_df['video'] == video
    ax.scatter(pca_df.loc[indicesToKeep, 'principal component 1'], pca_df.loc[indicesToKeep, 'principal component 2'],
               c=colors[video], marker=markers[video], s=5)

ax.legend(videos, bbox_to_anchor=(1.05, 1.0))
ax.grid()
plt.tight_layout()
plt.show()
