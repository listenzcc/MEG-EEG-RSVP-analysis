"""
File: 2.detect.erp.py
Author: Chuncheng Zhang
Date: 2026-01-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Detect the ERP from the RSVP data.

    1. Make the projector from the '3' event's -0.01, 0.01 second activity. 
    2. Remove the component of the projector.
    3. Remove the 10Hz waveform from every channel.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-30 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/step-1')
SUBJ = 'S05'
MODE = 'EEG'

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

OUTPUT_DIR = Path('./output/step-2') / f'{MODE}-{SUBJ}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-01-30 ------------------------
# Function and class


# 1. Make projector from the '3' event's -0.01 to 0.01 second activity


def create_event_projector(epochs, event_id='3', tmin=-0.01, tmax=0.01):
    """
    Create a projector from specific event activity in given time window
    """
    # Select only the event of interest
    epochs_event = epochs[event_id]

    # Extract the time window of interest
    event_data = epochs_event.copy().crop(tmin=tmin, tmax=tmax)

    # Average across the time window and epochs
    # Shape: (n_channels, n_times) -> average over time -> (n_channels,)
    event_avg = event_data.get_data().mean(axis=(0, 2))

    # Create a projector (orthogonal projection matrix)
    # P = I - u * u^T / (u^T * u) where u is our event template
    u = event_avg[:, np.newaxis]  # Make it column vector
    projector = np.eye(len(u)) - np.dot(u, u.T) / np.dot(u.T, u)

    return projector, event_avg

# 2. Remove the component of the projector


def apply_projector(data, projector):
    """
    Apply projector to remove specific component from data
    data shape: (n_epochs, n_channels, n_times)
    projector shape: (n_channels, n_channels)
    """
    n_epochs, n_channels, n_times = data.shape

    # Method 1: Apply projector to each epoch separately (more memory efficient)
    data_projected = np.zeros_like(data)
    for i in range(n_epochs):
        # For each epoch: (n_channels, n_times)
        epoch_data = data[i]  # shape: (n_channels, n_times)
        # Apply projector: (n_channels, n_channels) x (n_channels, n_times)
        data_projected[i] = np.dot(projector, epoch_data)

    # Alternative Method 2: Vectorized version
    # Reshape data to (n_epochs * n_channels, n_times) is WRONG for this operation
    # Instead, we need to think of it as applying projector to channel dimension

    return data_projected

# 3. Remove 10Hz waveform from every channel


def remove_10hz_component(epochs, freq_to_remove=10.0):
    """
    Remove specific frequency component using signal processing
    """
    # Method 1: Using notch filter (preferred for EEG data)
    # epochs_filtered = epochs.copy()
    # epochs_filtered.notch_filter(freqs=freq_to_remove, picks='all')

    # Alternative Method 2: Using FFT to remove specific frequency
    # This gives more control but is more complex
    def remove_frequency_fft(data, sfreq, freq_to_remove):
        n_epochs, n_channels, n_times = data.shape
        fft_data = np.fft.rfft(data, axis=2)
        freqs = np.fft.rfftfreq(n_times, 1/sfreq)

        # Find indices around the frequency to remove
        freq_indices = np.where((freqs >= freq_to_remove - 0.5) &
                                (freqs <= freq_to_remove + 0.5))[0]

        # Zero out these frequency components
        fft_data[:, :, freq_indices] = 0
        data_filtered = np.fft.irfft(fft_data, n=n_times, axis=2)
        return data_filtered

    data = remove_frequency_fft(
        epochs.get_data(), epochs.info['sfreq'], freq_to_remove)
    epochs_filtered = mne.EpochsArray(
        data, epochs.info, epochs.events, epochs.tmin, epochs.event_id)

    return epochs_filtered

# Main processing pipeline


def process_epochs(epochs):
    """
    Complete processing pipeline
    """
    # Create a copy to avoid modifying original data
    epochs_processed = epochs.copy()

    # 1. Create projector from event '3' in -0.01 to 0.01 second window
    projector, event_template = create_event_projector(
        epochs_processed,
        event_id='3',
        tmin=-0.01,
        tmax=0.01
    )

    print(f"Created projector with shape: {projector.shape}")
    print(f"Event template shape: {event_template.shape}")

    # 2. Apply projector to remove the component
    # Get the data, apply projector, and put back into epochs
    data = epochs_processed.get_data()
    data_projected = apply_projector(data, projector)

    # Update the epochs object with projected data
    epochs_processed._data = data_projected

    # 3. Remove 10Hz waveform from every channel
    for freq in tqdm([10, 20, 30, 40], 'Notching'):
        epochs_processed = remove_10hz_component(
            epochs_processed, freq_to_remove=freq)

    # Verify the processing
    print(f"\nProcessing complete:")
    print(f"- Original data shape: {data.shape}")
    print(f"- Projected data shape: {data_projected.shape}")
    print(f"- 10Hz component removed using notch filter")

    # You can also check the effect by plotting PSD
    # epochs_processed.plot_psd(fmin=5, fmax=15, average=True)

    return epochs_processed, projector


# %% ---- 2026-01-30 ------------------------
# Play ground
epochs = mne.read_epochs(DATA_DIR / f'{MODE}-{SUBJ}' / '1-3-epo.fif')

if MODE == 'EEG':
    montage = mne.channels.read_dig_fif('output/eeg-montage-dig.fif')
    montage.ch_names = [e.upper() for e in montage.ch_names]
    epochs.rename_channels({e: e.upper() for e in epochs.ch_names})
    epochs.set_montage(montage)


# %%

# %%
epochs_processed, projector = process_epochs(epochs)

print(epochs_processed)
print(projector)

epochs_processed.save(OUTPUT_DIR / f'1-3-epo.fif', overwrite=True)

for evt in ['1', '3']:
    evoked = epochs_processed[evt].average()
    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(OUTPUT_DIR / f'evoked-{evt}.png')
    plt.close(fig)


# %% ---- 2026-01-30 ------------------------
# Pending


# %% ---- 2026-01-30 ------------------------
# Pending
