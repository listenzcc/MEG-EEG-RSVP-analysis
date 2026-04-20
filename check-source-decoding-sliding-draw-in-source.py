"""
File: check-source-decoding-sliding-draw-in-source.py
Author: Chuncheng Zhang
Date: 2026-04-14
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check source decoding (sliding) results and draw in source.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-14 ------------------------
# Requirements and constants
import pickle
from mne.datasets import fetch_fsaverage
from util.easy_imports import *
from list_labels import read_average_labels

# %%
DATA_DIR = Path('./output/visual-source-decoding-sliding-auc')

# %% ---- 2026-04-14 ------------------------
# Function and class


def find_files():
    df = []
    files = sorted(DATA_DIR.rglob(f'scores_all_regions.dumps'))
    print(files)
    for p in files:
        mode, subj = p.parent.name.split('-')
        df.append({'mode': mode, 'subj': subj, 'path': p})
    df = pd.DataFrame(df)
    return df


def create_source_estimates(table, labels, mode, subjects_dir=None):
    """
    从table创建Peak和Delay的SourceEstimate对象并可视化
    """
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)

    # 读取源空间
    src = mne.read_source_spaces(os.path.join(
        subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    vertices = [src[0]['vertno'], src[1]['vertno']]
    n_verts = sum(len(v) for v in vertices)

    mode_data = table[table['mode'] == mode]

    # 初始化数据数组
    peak_data = np.zeros(n_verts)
    delay_data = np.zeros(n_verts)

    # 计算每个label的平均值并映射到顶点
    for label in labels:
        label_data = mode_data[mode_data['label'] == label.name]

        if len(label_data) == 0:
            raise ValueError('No data error')
            continue

        avg_peak = label_data['peak'].mean()
        avg_delay = label_data['delay'].mean()

        # 确定半球
        hemi_idx = 0 if label.hemi == 'lh' else 1

        # 找到顶点索引
        vert_indices = np.where(
            np.isin(vertices[hemi_idx], label.vertices))[0]
        if hemi_idx == 1:
            vert_indices += len(vertices[0])

        if len(vert_indices) > 0:
            peak_data[vert_indices] = avg_peak
            delay_data[vert_indices] = avg_delay

    # 创建STC对象
    peak_stc = mne.SourceEstimate(
        peak_data.reshape(-1, 1),
        vertices=vertices,
        tmin=0,
        tstep=1,
        subject='fsaverage'
    )

    delay_stc = mne.SourceEstimate(
        delay_data.reshape(-1, 1),
        vertices=vertices,
        tmin=0,
        tstep=1,
        subject='fsaverage'
    )

    # 直接plot
    print(f"\n{mode} - Peak AUC:")
    brain_peak = peak_stc.plot(
        subject='fsaverage',
        surface='inflated',
        hemi='split',
        colormap='hot',
        views=['lateral', 'medial'],
        subjects_dir=subjects_dir,
        # clim=dict(kind='percent', lims=[50, 75, 95]),
        clim=dict(kind='value', lims=[0.5, 0.7, 0.9]),
        time_label=f'Peak AUC {mode}',
        size=(800, 400)
    )

    print(f"\n{mode} - Delay:")
    brain_delay = delay_stc.plot(
        subject='fsaverage',
        surface='inflated',
        hemi='split',
        colormap='viridis',
        views=['lateral', 'medial'],
        subjects_dir=subjects_dir,
        # clim=dict(kind='percent', lims=[20, 50, 80]),
        clim=dict(kind='value', lims=[0.2, 0.4, 0.6]),
        time_label=f'Delay (s) {mode}',
        size=(800, 400)
    )

    return peak_stc, delay_stc


def create_temporal_source_estimates(table, labels, times, mode, subjects_dir=None):
    """
    创建包含完整时间动态的SourceEstimate对象

    Parameters:
    -----------
    table : DataFrame
        包含scores列的数据框
    labels : list
        mne Label对象列表
    times : array
        时间点数组
    mode : str
        EEG | MEG
    subjects_dir : str
        fsaverage subjects目录
    """
    if subjects_dir is None:
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = os.path.dirname(fs_dir)

    # 读取源空间
    src = mne.read_source_spaces(os.path.join(
        subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    vertices = [src[0]['vertno'], src[1]['vertno']]
    n_verts = sum(len(v) for v in vertices)
    n_times = len(times)

    mode_data = table[table['mode'] == mode]

    # 初始化数据数组: (n_verts, n_times)
    temporal_data = np.zeros((n_verts, n_times))

    # 对每个label，计算平均时间序列并映射到顶点
    for label in labels:
        label_data = mode_data[mode_data['label'] == label.name]
        if len(label_data) == 0:
            continue

        # 收集所有被试的时间序列
        all_scores = np.array(
            [scores for scores in label_data['scores'].values])

        # 计算平均时间序列
        avg_timeseries = np.mean(all_scores, axis=0)

        # 确定半球
        hemi_idx = 0 if label.hemi == 'lh' else 1

        # 找到顶点索引
        vert_indices = np.where(np.isin(vertices[hemi_idx], label.vertices))[0]
        if hemi_idx == 1:
            vert_indices += len(vertices[0])

        if len(vert_indices) > 0:
            # 将时间序列赋值给该label的所有顶点
            temporal_data[vert_indices, :] = avg_timeseries

    # 创建STC对象
    stc = mne.SourceEstimate(
        temporal_data,
        vertices=vertices,
        tmin=times[0],
        tstep=times[1] - times[0],
        subject='fsaverage'
    )

    print(f"\n{mode} - Temporal dynamics (shape: {stc.data.shape})")

    clim = dict(kind='value', lims=[0.5, 0.65, 0.8])
    colormap = 'hot'
    if mode == 'sub':
        clim = 'auto'
        colormap = 'RdBu'

    # Plot temporal dynamics
    brain = stc.plot(
        subject='fsaverage',
        surface='inflated',
        hemi='split',
        colormap=colormap,
        views=['lateral', 'medial'],
        subjects_dir=subjects_dir,
        # clim=dict(kind='percent', lims=[50, 75, 95]),
        clim=clim,
        time_label=f'AUC {mode}',
        title=f'{mode} - Temporal Dynamics',
        size=(800, 400),
        initial_time=0.1,  # 初始显示的时间点
    )

    return stc


def add_subtraction_mode(table):
    """
    为每个被试和脑区计算 MEG - EEG 的差异，添加为新的 mode='sub'

    Parameters:
    -----------
    table : DataFrame
        包含 mode, subj, label, peak, delay, scores 等列的原始数据框

    Returns:
    --------
    DataFrame
        添加了 mode='sub' 记录的完整数据框
    """
    # 分别获取 MEG 和 EEG 数据
    meg_data = table[table['mode'] == 'MEG'].copy()
    eeg_data = table[table['mode'] == 'EEG'].copy()

    # 设置索引以便对齐
    meg_data = meg_data.set_index(['subj', 'label'])
    eeg_data = eeg_data.set_index(['subj', 'label'])

    # 找到共同的 subject-label 组合
    common_indices = meg_data.index.intersection(eeg_data.index)

    # 创建减法结果
    sub_records = []
    for idx in common_indices:
        meg_row = meg_data.loc[idx]
        eeg_row = eeg_data.loc[idx]

        # 计算差异
        sub_record = {
            'mode': 'sub',  # MEG - EEG
            'subj': idx[0],
            'label': idx[1],
            'peak': meg_row['peak'] - eeg_row['peak'],
            'delay': meg_row['delay'] - eeg_row['delay'],
            'scores': meg_row['scores'] - eeg_row['scores'],  # 时间序列相减
            'path': None  # 新记录没有原始路径
        }
        sub_records.append(sub_record)

    # 将新记录添加到原table
    sub_df = pd.DataFrame(sub_records)
    new_table = pd.concat([table, sub_df], ignore_index=True)

    return new_table


# %% ---- 2026-04-14 ------------------------
# Play ground
labels = read_average_labels('PALS_B12_Visuotopic')
files_table = find_files()
files_table

# %%
table = []
times = np.linspace(-0.5, 1.5, 401)
for i, row in tqdm(files_table.iterrows(), total=len(files_table)):
    p = row['path']
    obj = pickle.load(open(p, 'rb'))
    for label, array in obj.items():
        data = np.mean(array, axis=0)
        peak = np.max(data)
        delay = times[np.argmax(data)]
        table.append({
            'mode': row['mode'],  # Mode: MEG | EEG
            'subj': row['subj'],  # Subject: S01 | S02 | ...
            'label': label.name,  # Brain area label
            'peak': peak,  # Peak AUC
            'delay': delay,  # Delay of peak AUC
            'scores': data  # time series of AUC
        })

table = pd.DataFrame(table)
table = add_subtraction_mode(table)
table

# %%
# 使用
if False:
    peak_stc, delay_stc = create_source_estimates(table, labels, 'EEG')
    peak_stc, delay_stc = create_source_estimates(table, labels, 'MEG')
    input('Press enter to escape.')

# %%
stc = create_temporal_source_estimates(table, labels, times, 'EEG')
stc = create_temporal_source_estimates(table, labels, times, 'MEG')
stc = create_temporal_source_estimates(table, labels, times, 'sub')
input('Press enter to escape.')

# %%


# %% ---- 2026-04-14 ------------------------
# Pending


# %% ---- 2026-04-14 ------------------------
# Pending
