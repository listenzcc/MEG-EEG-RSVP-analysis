"""
File: list_labels.py
Author: Chuncheng Zhang
Date: 2026-04-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    List the labels info.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-15 ------------------------
# Requirements and constants
from util.easy_imports import *
from mne.datasets import fetch_fsaverage

# %% ---- 2026-04-15 ------------------------
# Function and class


def read_average_labels(parc='aparc.a2009s'):
    # 1. 准备模板数据（使用fsaverage模板）
    print("下载fsaverage模板...")
    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    labels = mne.read_labels_from_annot(
        subject='fsaverage',
        parc=parc,
        subjects_dir=subjects_dir
    )

    # 2. Setup labels differently for every parc
    if parc == 'aparc.a2009s':
        labels = labels[:-2]
    elif parc == 'PALS_B12_Visuotopic':
        labels = [e for e in labels if e.name.startswith('Visuotopic')]
    else:
        raise ValueError(f'The parc is unknown: {parc=}')

    return labels


# %% ---- 2026-04-15 ------------------------
# Play ground
if __name__ == '__main__':
    labels = read_average_labels('aparc.a2009s')
    print(labels)

    labels = read_average_labels('PALS_B12_Visuotopic')
    print(labels)


# %% ---- 2026-04-15 ------------------------
# Pending


# %% ---- 2026-04-15 ------------------------
# Pending
