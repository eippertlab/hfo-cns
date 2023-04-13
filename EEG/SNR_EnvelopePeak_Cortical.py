# Plot single subject envelopes with bounds where peak should be
# Calculate SNR and add information to plot


import os
import mne
import numpy as np
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
    plot_image = True
    save_to_excel = False
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

    figure_path = '/data/p_02718/Images/CCA_eeg/SNR&EnvelopePeak/'
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
            fname = f"{freq_band}_{cond_name}.fif"
            input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
            epochs = mne.read_epochs(input_path + fname, preload=True)

            if cond_name == 'median':
                sep_latency = df_timing.loc[subject, f"N20"]
            elif cond_name == 'tibial':
                sep_latency = df_timing.loc[subject, f"P39"]

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
                signal_window = 7.5/1000
                snr = calculate_snr(evoked.copy(), noise_window, signal_window, sep_latency)
                snr_cond[subject-1][c] = snr

                # Get Envelope
                envelope = evoked.copy().apply_hilbert(envelope=True)
                data = envelope.get_data()

                if plot_image:
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
                    # Add lines for mean+-3*std of noise period
                    noise_data = evoked.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data().reshape(-1)
                    noise_mean = np.mean(noise_data)
                    noise_std = np.std(noise_data)
                    ax[c].axhline(y=noise_mean - 3 * noise_std, color='blue', linewidth='1')
                    ax[c].axhline(y=noise_mean + 3 * noise_std, color='blue', linewidth='1')
                    plt.suptitle(f'Subject {subject}, {trigger_name}')
                    plt.tight_layout()
                    plt.savefig(figure_path+f'{subject_id}_{cond_name}')

        if save_to_excel:
            # Create data frame
            if cond_name == 'median':
                df_med = pd.DataFrame(snr_cond, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4'])
            else:
                df_tib = pd.DataFrame(snr_cond, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4'])

    if save_to_excel:
        with pd.ExcelWriter(f'{figure_path}ComponentSNR.xlsx') as writer:
            df_med.to_excel(writer, sheet_name='Median Stimulation')
            df_tib.to_excel(writer, sheet_name='Tibial Stimulation')
