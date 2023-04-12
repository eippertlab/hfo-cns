# Plot single subject envelopes with bounds where peak should be
# Calculate SNR and add information to plot
# While we're running this, create an excel sheet with the SNR for each subject and component


import os
import mne
import numpy as np
from scipy.io import loadmat
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.calculate_snr import calculate_snr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    cca_good = False  # Use CCA performed only on good trials when true, otherwise all trials
    create_plot = False
    subjects = np.arange(1, 37)
    # subjects = np.arange(1, 2)
    conditions = [2, 3]
    freq_band = 'sigma'
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/Cortical_Timing.xlsx')
    df_timing = pd.read_excel(xls_timing, 'Timing')
    df_timing.set_index('Subject', inplace=True)

    n_components = 4

    if cca_good:
        figure_path = '/data/p_02718/Images/CCA_good/SNR&EnvelopePeak/'
        os.makedirs(figure_path, exist_ok=True)
    else:
        figure_path = '/data/p_02718/Images/CCA/SNR&EnvelopePeak/'
        os.makedirs(figure_path, exist_ok=True)

    for condition in conditions:
        snr_cond = [[0 for x in range(n_components)] for x in range(len(subjects))]

        # Set variables
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        for subject in subjects:
            fig, ax = plt.subplots(2, 2)
            ax = ax.flatten()
            subject_id = f'sub-{str(subject).zfill(3)}'
            # Select the right files
            fname = f"{freq_band}_{cond_name}.fif"
            if cca_good:
                input_path = "/data/pt_02718/tmp_data/cca_goodonly/" + subject_id + "/"
            else:
                input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
            epochs = mne.read_epochs(input_path + fname, preload=True)

            potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
            fname_pot = 'potential_latency.mat'
            matdata = loadmat(potential_path + fname_pot)
            if cond_name == 'median':
                sep_latency = matdata['med_potlatency'][0][0]/1000
            elif cond_name == 'tibial':
                sep_latency = matdata['tib_potlatency'][0][0]/1000

            for c in np.arange(0, n_components):  # Loop through all components
                # Need to pick channel
                channel = f'Cor{c+1}'
                epochs_ch = epochs.copy().pick_channels([channel])

                if cond_name == 'median':
                    evoked = epochs_ch.crop(tmin=-0.1, tmax=0.05).copy().average()
                elif cond_name == 'tibial':
                    evoked = epochs_ch.crop(tmin=-0.1, tmax=0.07).copy().average()

                # # Get SNR of HFO
                noise_window = [-100/1000, -10/1000]
                signal_window = 5/1000
                snr = calculate_snr(evoked.copy(), noise_window, signal_window, sep_latency)
                snr_cond[subject-1][c] = snr

                # Get Envelope
                envelope = evoked.copy().apply_hilbert(envelope=True)
                data = envelope.get_data()

                if create_plot:
                    # Plot Envelope with SNR
                    ax[c].plot(evoked.times, evoked.get_data().reshape(-1), color='lightblue')
                    ax[c].plot(evoked.times, data.reshape(-1))
                    ax[c].set_xlabel('Time (s)')
                    ax[c].set_ylabel('Amplitude')
                    ax[c].axvline(x=sep_latency-signal_window, color='r', linewidth='1')
                    ax[c].axvline(x=sep_latency, color='green', linewidth='1')
                    ax[c].axvline(x=sep_latency+signal_window, color='r', linewidth='1')
                    ax[c].set_title(f'Component {c+1}, SNR: {snr:.2f}')
                    if cond_name == 'median':
                        ax[c].set_xlim([0.0, 0.05])
                    else:
                        ax[c].set_xlim([0.0, 0.07])
                    plt.suptitle(f'Subject {subject}, {trigger_name}')
                    plt.tight_layout()
                    plt.savefig(figure_path+f'{subject_id}_{cond_name}')

        # Create data frame
        if cond_name == 'median':
            df_med = pd.DataFrame(snr_cond, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4'])
        else:
            df_tib = pd.DataFrame(snr_cond, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4'])

    with pd.ExcelWriter(f'{figure_path}ComponentSNR.xlsx') as writer:
        df_med.to_excel(writer, sheet_name='Median Stimulation')
        df_tib.to_excel(writer, sheet_name='Tibial Stimulation')
