"""
File: decoding.py
Author: Chuncheng Zhang
Date: 2026-03-31
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding for the RSVP data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-31 ------------------------
# Requirements and constants
from mne.preprocessing import Xdawn
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from util.easy_imports import *

# %%
SUBJ = 'S02'
MODE = 'EEG'

# if len(sys.argv) > 1:
#     _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
DATA_DIR = Path(f'output/decoding-step-1/{MODE}-{SUBJ}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

# %%
OUTPUT_DIR = Path(f'output/decoding/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-03-31 ------------------------
# Function and class


# %% ---- 2026-03-31 ------------------------
# Play ground
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)
epochs3 = mne.read_epochs(DATA_DIR / 'epochs-3-epo.fif', preload=True)
epochs4 = mne.read_epochs(DATA_DIR / 'epochs-4-epo.fif', preload=True)

print(epochs1, epochs2, epochs3, epochs4)

# %% ---- 2026-03-31 ------------------------
# Pending

# 合并所有epochs数据
epochs_all = mne.concatenate_epochs([
    epochs1,
    epochs2,
    epochs3,
    # epochs4
])

# 获取数据X和标签y
X = epochs_all.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # 标签 1,2,3,4

# 10折交叉验证
cv = StratifiedKFold(n_splits=10, shuffle=True)

# 构建pipeline: XDawn + PCA降维 + SVM
pipeline = make_pipeline(
    Xdawn(n_components=4),  # , estimator='scm'),  # 特征提取
    StandardScaler(),
    PCA(n_components=0.95),  # 保留95%方差
    SVC(kernel='rbf', decision_function_shape='ovr')
)

# 执行交叉验证
scores = cross_val_score(pipeline, epochs_all, y, cv=cv,
                         scoring='accuracy', n_jobs=-1)

print(f"10折交叉验证准确率: {scores}")
print(f"平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# %% ---- 2026-03-31 ------------------------
# Pending
