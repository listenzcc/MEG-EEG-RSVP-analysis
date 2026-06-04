"""
File: quick-slow.py
Author: Chuncheng Zhang
Date: 2026-06-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot the quick - slow source.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-03 ------------------------
# Requirements and constants
from util.easy_imports import *
from mne import read_source_estimate


# %% ---- 2026-06-03 ------------------------
# Function and class


# %% ---- 2026-06-03 ------------------------
# Play ground
stc_quick = read_source_estimate('./output/quick.stc')
stc_slow = read_source_estimate('./output/slow.stc')
stc = stc_quick - stc_slow
stc.subject = 'fsaverage'

stc.plot(title='Quick - Slow Source Estimate')
input('Press Enter to continue...')

# %% ---- 2026-06-03 ------------------------
# Pending


# %% ---- 2026-06-03 ------------------------
# Pending
