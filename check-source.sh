#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

# --------------------------------------------------------------------------------
python source-estimation-on-evoked.py --mode MEG --file ./output/artificial-by-diff-times/MEG-evoked-slow-by-diff-times-ave.fif --output ./output/slow.stc &
python source-estimation-on-evoked.py --mode MEG --file ./output/artificial-by-diff-times/MEG-evoked-quick-by-diff-times-ave.fif --output ./output/quick.stc &
# python source-estimation-on-evoked.py --mode MEG --file ./output/step-1-subjects-average/MEG/1-epo-ave.fif &
# python source-estimation-on-evoked.py --mode MEG --file ./output/step-1-subjects-average-proj/MEG/1-withproj-epo-ave.fif &
