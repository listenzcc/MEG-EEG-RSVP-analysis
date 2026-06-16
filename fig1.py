'''
ERP
'''

# %%
from util.easy_imports import *


# %%
eeg_evoked_t = mne.read_evokeds(
    Path('./output/step-1-subjects-average/EEG/1-epo-ave.fif'))[0]
display(eeg_evoked_t)

eeg_evoked_tp = mne.read_evokeds(
    Path('./output/step-1-subjects-average-proj/EEG/1-withproj-epo-ave.fif'))[0]
display(eeg_evoked_tp)

eeg_evoked_nt = mne.read_evokeds(
    Path('./output/step-1-subjects-average/EEG/2-epo-ave.fif'))[0]
display(eeg_evoked_nt)

eeg_evoked_k = mne.read_evokeds(
    Path('./output/step-1-subjects-average/EEG/3-epo-ave.fif'))[0]
display(eeg_evoked_k)

meg_evoked_t = mne.read_evokeds(
    Path('./output/step-1-subjects-average/MEG/1-epo-ave.fif'))[0]
display(meg_evoked_t)

meg_evoked_tp = mne.read_evokeds(
    Path('./output/step-1-subjects-average-proj/MEG/1-withproj-epo-ave.fif'))[0]
display(meg_evoked_tp)

meg_evoked_nt = mne.read_evokeds(
    Path('./output/step-1-subjects-average/MEG/2-epo-ave.fif'))[0]
display(meg_evoked_nt)

meg_evoked_k = mne.read_evokeds(
    Path('./output/step-1-subjects-average/MEG/3-epo-ave.fif'))[0]
display(meg_evoked_k)


# %%
data = [
    dict(
        evoked=eeg_evoked_t.copy(),
        mark='target',
        mode='EEG',
        name='EEG-target'
    ),
    dict(
        evoked=eeg_evoked_tp.copy(),
        mark='target',
        mode='EEG',
        name='EEG-target-proj'
    ),
    dict(
        evoked=eeg_evoked_t.copy(),
        mark='nontarget',
        mode='EEG',
        name='EEG-nontarget'
    ),
    dict(
        evoked=meg_evoked_t.copy(),
        mark='target',
        mode='MEG',
        name='MEG-target'
    ),
    dict(
        evoked=meg_evoked_tp.copy(),
        mark='target',
        mode='MEG',
        name='MEG-target-proj'
    ),
    dict(
        evoked=meg_evoked_t.copy(),
        mark='nontarget',
        mode='MEG',
        name='MEG-nontarget'
    ),
]

for d in data:
    evoked = d['evoked']
    mark = d['mark']
    mode = d['mode']
    name = d['name']

    # if mark == 'target':
    #     evoked.filter(l_freq=None, h_freq=8, n_jobs=-1)

    if mark == 'nontarget':
        evoked.filter(l_freq=8, h_freq=12, n_jobs=-1)

    evoked.crop(tmin=-0.2, tmax=1.2)

    fig = evoked.plot(spatial_colors=False, show=False)

    red_ch_name = 'oz' if mode == 'EEG' else 'mzo02'

    # Get the axes and modify the lines
    axes = fig.get_axes()
    for ax in axes:
        lines = ax.get_lines()
        for line, ch_name in zip(lines, evoked.ch_names):
            if red_ch_name == ch_name.lower():
                line.set_color('red')
                line.set_linewidth(2.0)
            else:
                line.set_color('#aaaaaa')
        ax.axvline(0, linestyle='--', color='gray')

        if mode == 'MEG' and mark == 'target':
            ax.axvline(0.23, linestyle='--', color='blue')
            ax.axvline(0.35, linestyle='--', color='blue')

        if mode == 'EEG' and mark == 'target':
            ax.axvline(0.23, linestyle='--', color='blue')
            ax.axvline(0.30, linestyle='--', color='blue')
            ax.axvline(0.35, linestyle='--', color='blue')

    fig.savefig(f'./collect-results/fig1/{name}.png')

# plt.show()

# %%
meg_evoked_tp.plot_topomap(times=[0.23, 0.3, 0.35], ch_type='mag', show=False)
# plt.show()
plt.savefig(f'./collect-results/fig1/MEG-topo.png')

eeg_evoked_tp.plot_topomap(times=[0.23, 0.3, 0.35], ch_type='eeg', show=False)
# plt.show()
plt.savefig(f'./collect-results/fig1/EEG-topo.png')


# %%

# %%
# evoked = meg_evoked_t

# fig = evoked.plot(spatial_colors=False, show=False)

# # Get the axes and modify the lines
# axes = fig.get_axes()
# for ax in axes:
#     lines = ax.get_lines()
#     for line, ch_name in zip(lines, evoked.ch_names):
#         if 'mzo02' == ch_name.lower():
#             line.set_color('red')
#             line.set_linewidth(2.0)
#         elif '-mlo43' == ch_name.lower():
#             line.set_color('green')
#             line.set_linewidth(2.0)
#         elif '-mro43' == ch_name.lower():
#             line.set_color('blue')
#             line.set_linewidth(2.0)
#         else:
#             line.set_color('gray')
#     ax.axvline(0, linestyle='--', color='gray')

# plt.show()

# %%

# %%
