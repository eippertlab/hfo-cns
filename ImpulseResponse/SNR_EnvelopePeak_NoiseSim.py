# Plot single subject envelopes with bounds where peak should be
# Calculate SNR and add information to plot
# Implement automatic selection of components
# This will select components BUT you still need to manually choose flipping of components


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.calculate_snr_hfo import calculate_snr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
from Common_Functions.check_excel_exist_component import check_excel_exist
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    save_to_excel = True  # If we want to save the SNR values on each run
    scaler = 12  # Identifier for how much we multiplied standard noise (std ~1 by)

    runs = np.arange(0, 100)  # 0 through 99 to access simulated subject data
    input_path = f"/data/pt_02718/tmp_data/noise_simulations_{scaler}timesnoise/"

    snr_threshold = 5
    snr_allruns = []

    for run in runs:
        run_trials = np.load(f"{input_path}{run}_filtered.npy")
        # Get SNR of  simulated
        noise_period_filtered = [x[10:461] for x in run_trials]
        std_postfilter = np.std(noise_period_filtered)
        # Want peak 5ms before and 5ms after (5e-3 times sf of 5000 is 25 samples)
        sig_amp = np.max(np.mean(run_trials, axis=0)[1000-25:1000+25])
        snr = sig_amp/std_postfilter
        snr_allruns.append(snr)

    np.savetxt(f"{input_path}SNR_filtered", snr_allruns, fmt='%f')
