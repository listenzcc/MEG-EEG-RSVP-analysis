#!/usr/bin/env zsh

source ~/.zshrc
conda activate mne-analysis

echo -----------------------------
echo python env
which python
python --version

mode=EEG

script=1.1.plot.evoked.py

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

script=1.1.a.plot.evoked.all.subjects.py

python $script $mode

mode=MEG

script=1.1.plot.evoked.py

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

script=1.1.a.plot.evoked.all.subjects.py

python $script $mode

# script=2.2.detect.erp.py

# python $script S01 $mode &
# python $script S02 $mode &
# python $script S03 $mode &
# python $script S04 $mode &
# python $script S05 $mode &
# python $script S06 $mode &
# python $script S07 $mode &
# python $script S08 $mode &
# python $script S09 $mode &
# python $script S10 $mode

# script=3.concat.subjects.py

# python $script $mode

# script=4.erp.of.roi.channels.py

# python $script $mode

# mode=MEG
# python $script S01 $mode &
# python $script S02 $mode &
# python $script S03 $mode &
# python $script S04 $mode &
# python $script S05 $mode &
# python $script S06 $mode &
# python $script S07 $mode &
# python $script S08 $mode &
# python $script S09 $mode &
# python $script S10 $mode
