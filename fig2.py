'''
ERP with quick or slow response
'''

# %%
from util.easy_imports import *

# %%
eeg_evoked_tp_quick = mne.read_evokeds(
    Path('./output/artificial-by-diff-times/EEG-evoked-proj-quick-by-diff-times-ave.fif')
)[0]
display(eeg_evoked_tp_quick)

eeg_evoked_tp_slow = mne.read_evokeds(
    Path('./output/artificial-by-diff-times/EEG-evoked-proj-slow-by-diff-times-ave.fif')
)[0]
display(eeg_evoked_tp_slow)

meg_evoked_tp_quick = mne.read_evokeds(
    Path('./output/artificial-by-diff-times/MEG-evoked-proj-quick-by-diff-times-ave.fif')
)[0]
display(meg_evoked_tp_quick)

meg_evoked_tp_slow = mne.read_evokeds(
    Path('./output/artificial-by-diff-times/MEG-evoked-proj-slow-by-diff-times-ave.fif')
)[0]
display(meg_evoked_tp_slow)

# %%

data = [
    dict(
        evoked=eeg_evoked_tp_quick.copy(),
        mode='EEG',
        name='EEG-quick'
    ),
    dict(
        evoked=eeg_evoked_tp_slow.copy(),
        mode='EEG',
        name='EEG-slow'
    ),
    dict(
        evoked=meg_evoked_tp_quick.copy(),
        mode='MEG',
        name='MEG-quick'
    ),
    dict(
        evoked=meg_evoked_tp_slow.copy(),
        mode='MEG',
        name='MEG-slow'
    ),
]

for d in data:
    evoked = d['evoked']
    mode = d['mode']
    name = d['name']

    # evoked.filter(l_freq=None, h_freq=8, n_jobs=-1)
    evoked.crop(tmin=-0.2, tmax=1.2)

    fig = evoked.plot(spatial_colors=False, show=False)

    red_ch_name = 'oz' if mode == 'EEG' else 'mzo02'  # 'mzo02'

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

        if mode == 'MEG':
            ax.axvline(0.22, linestyle='--', color='blue')
            ax.axvline(0.24, linestyle='--', color='blue')
            ax.axvline(0.35, linestyle='--', color='blue')

        if mode == 'EEG':
            ax.axvline(0.16, linestyle='--', color='blue')
            ax.axvline(0.30, linestyle='--', color='blue')
            ax.axvline(0.35, linestyle='--', color='blue')
            ax.axvline(0.46, linestyle='--', color='blue')

    fig.suptitle(name)

    fig.savefig(f'./collect-results/fig2/{name}.png')

plt.show()

# %%
meg_evoked_tp_quick.plot_topomap(
    times=[0.22, 0.24, 0.35], ch_type='mag', show=False)
plt.suptitle('MEG-quick')
plt.savefig(f'./collect-results/fig2/MEG-quick-topo.png')
meg_evoked_tp_slow.plot_topomap(
    times=[0.22, 0.24, 0.35], ch_type='mag', show=False)
plt.suptitle('MEG-slow')
plt.savefig(f'./collect-results/fig2/MEG-slow-topo.png')

eeg_evoked_tp_quick.plot_topomap(
    times=[0.16, 0.30, 0.35, 0.46], ch_type='eeg', show=False)
plt.suptitle('EEG-quick')
plt.savefig(f'./collect-results/fig2/EEG-quick-topo.png')
eeg_evoked_tp_slow.plot_topomap(
    times=[0.16, 0.30, 0.35, 0.46], ch_type='eeg', show=False)
plt.suptitle('EEG-slow')
plt.savefig(f'./collect-results/fig2/EEG-slow-topo.png')

# %%
meg_evoked_tp_quick.plot_joint()
meg_evoked_tp_slow.plot_joint()
0
# %%
