#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

# script=pipeline/decoding.1.raw.to.epochs.py
# script=source-decoding-sliding-auc.py
# script=decoding-trial.py
script=decoding-trial-124.py

mode=EEG
python $script --subj S01 --mode $mode 
python $script --subj S02 --mode $mode 
python $script --subj S03 --mode $mode 
python $script --subj S04 --mode $mode 
python $script --subj S05 --mode $mode 
python $script --subj S06 --mode $mode 
python $script --subj S07 --mode $mode 
python $script --subj S08 --mode $mode 
python $script --subj S09 --mode $mode 
python $script --subj S10 --mode $mode

mode=MEG
python $script --subj S01 --mode $mode 
python $script --subj S02 --mode $mode 
python $script --subj S03 --mode $mode 
python $script --subj S04 --mode $mode 
python $script --subj S05 --mode $mode 
python $script --subj S06 --mode $mode 
python $script --subj S07 --mode $mode 
python $script --subj S08 --mode $mode 
python $script --subj S09 --mode $mode 
python $script --subj S10 --mode $mode
