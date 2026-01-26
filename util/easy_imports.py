# %%
import mne
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from pathlib import Path
from loguru import logger
from IPython.display import display

# %%
n_jobs = 32
logger.add('log/MEG-EEG-RSVP.log', rotation='1 MB')

# %%
