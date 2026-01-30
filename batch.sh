#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

# script=1.plot.evoked.py
script=2.detect.erp.py
mode=EEG

python $script S01 $mode &
python $script S02 $mode &
python $script S03 $mode &
python $script S04 $mode &
python $script S05 $mode &
python $script S06 $mode &
python $script S07 $mode &
python $script S08 $mode &
python $script S09 $mode &
python $script S10 $mode

mode=MEG
python $script S01 $mode &
python $script S02 $mode &
python $script S03 $mode &
python $script S04 $mode &
python $script S05 $mode &
python $script S06 $mode &
python $script S07 $mode &
python $script S08 $mode &
python $script S09 $mode &
python $script S10 $mode