#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis
# conda activate braindecode

echo -----------------------------
echo python env
which python
python --version

# script=pipeline/decoding.1.raw.to.epochs.py
# script=source-decoding-sliding-auc.py
# script=decoding-trial.py
# script=decoding-trial-124.py
# script=decoding-sliding-auc-remove-artificial.py
script=./decoding-sliding-with-quick-slow-trials.py

# It requires braindecode env
# script=shallow-cnn-decoding.py

conditions=(quick slow)
modes=(EEG MEG)
modes=(EEG)
subjects=(S01 S02 S03 S04 S05 S06 S07 S08 S09 S10)

for mode in "${modes[@]}"; do
    for subj in "${subjects[@]}"; do
        for cond in "${conditions[@]}"; do
            python $script --subj $subj --mode $mode --cond $cond
        done
    done
done

# cond=quick
#
# mode=EEG
# python $script --subj S01 --mode $mode --cond $cond
# python $script --subj S02 --mode $mode --cond $cond
# python $script --subj S03 --mode $mode --cond $cond
# python $script --subj S04 --mode $mode --cond $cond
# python $script --subj S05 --mode $mode --cond $cond
# python $script --subj S06 --mode $mode --cond $cond
# python $script --subj S07 --mode $mode --cond $cond
# python $script --subj S08 --mode $mode --cond $cond
# python $script --subj S09 --mode $mode --cond $cond
# python $script --subj S10 --mode $mode --cond $cond
#
# mode=MEG
# python $script --subj S01 --mode $mode --cond $cond
# python $script --subj S02 --mode $mode --cond $cond
# python $script --subj S03 --mode $mode --cond $cond
# python $script --subj S04 --mode $mode --cond $cond
# python $script --subj S05 --mode $mode --cond $cond
# python $script --subj S06 --mode $mode --cond $cond
# python $script --subj S07 --mode $mode --cond $cond
# python $script --subj S08 --mode $mode --cond $cond
# python $script --subj S09 --mode $mode --cond $cond
# python $script --subj S10 --mode $mode --cond $cond
