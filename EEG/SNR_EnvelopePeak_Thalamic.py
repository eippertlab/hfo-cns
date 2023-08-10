# Plot single subject envelopes with bounds where peak should be
# Calculate SNR and add information to plot
# Implement automatic selection of components
# This will select components BUT you still need to manually choose flipping of components


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
    save_to_excel = True  # If we want to save the SNR values on each run

    freq_band = 'sigma'
    srmr_nr = 1  # Experiment 2 not implemented for this part

    if srmr_nr != 1:
        print('Error: Not implemented for srmr_nr 2')
        exit()

    if srmr_nr == 1:
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [2, 3]  # Conditions of interest
        # xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/Cortical_Timing.xlsx')
        component_fname = '/data/pt_02718/tmp_data/Components_EEG_Thalamic_Updated.xlsx'
        visibility_fname = '/data/pt_02718/tmp_data/Visibility_Thalamic_Updated.xlsx'
        figure_path = '/data/p_02718/Images/CCA_eeg_thalamic/SNR&EnvelopePeak/'
        os.makedirs(figure_path, exist_ok=True)

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
        conditions = [3, 5]  # Conditions of interest - med_mixed and tib_mixed [3, 5]
        # xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/Cortical_Timing.xlsx')
        component_fname = '/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated.xlsx'
        visibility_fname = '/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated.xlsx'
        figure_path = '/data/p_02718/Images_2/CCA_eeg_thalamic/SNR&EnvelopePeak/'
        os.makedirs(figure_path, exist_ok=True)

    # Check the component file is already generated - want to store the flipping info in the same place so easier to
    # do it this way
    component_sheetname = 'CCA'
    visibility_sheetname = 'CCA_Brain'
    col_names = ['Subject', 'sigma_median_comp', 'sigma_median_flip', 'sigma_tibial_comp', 'sigma_tibial_flip']
    if not os.path.isfile(component_fname):
        print(f'Created an excel file with the name {component_fname} and the column headers '
              f'{col_names} '
              f'before continuing')
        exit()

    # df_timing = pd.read_excel(xls_timing, 'Timing')
    # df_timing.set_index('Subject', inplace=True)

    df_comp = pd.read_excel(component_fname, component_sheetname)
    df_comp.set_index('Subject', inplace=True)

    df_vis = pd.read_excel(visibility_fname, visibility_sheetname)
    df_vis.set_index('Subject', inplace=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    n_components = 4
    snr_threshold = 5

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
            if srmr_nr == 1:
                fname = f"{freq_band}_{cond_name}.fif"
                input_path = "/data/pt_02718/tmp_data/cca_eeg_thalamic/" + subject_id + "/"
            elif srmr_nr == 2:
                fname = f"{freq_band}_{cond_name}.fif"
                input_path = "/data/pt_02718/tmp_data_2/cca_eeg_thalamic/" + subject_id + "/"
            epochs = mne.read_epochs(input_path + fname, preload=True)

            if cond_name in ['median', 'med_mixed']:
                sep_latency = 0.013
            elif cond_name in ['tibial', 'tib_mixed']:
                sep_latency = 0.030

            snr_comp = []
            peak_latency_comp = []
            for c in np.arange(0, n_components):  # Loop through all components

                # Need to pick channel
                channel = f'Cor{c+1}'
                epochs_ch = epochs.copy().pick_channels([channel])

                if cond_name in ['median', 'med_mixed']:
                    evoked = epochs_ch.crop(tmin=-0.1, tmax=0.05).copy().average()
                elif cond_name in ['tibial', 'tib_mixed']:
                    evoked = epochs_ch.crop(tmin=-0.1, tmax=0.07).copy().average()

                # # Get SNR of HFO
                noise_window = [-100/1000, -10/1000]
                signal_window = 3/1000
                snr = calculate_snr(evoked.copy(), noise_window, signal_window, sep_latency)
                snr_comp.append(snr)
                snr_cond[subject-1][c] = snr

                # Get Envelope
                envelope = evoked.copy().apply_hilbert(envelope=True)
                data = envelope.get_data()
                if cond_name in ['median', 'med_mixed']:
                    ch_name, latency = envelope.get_peak(tmin=0, tmax=50/1000, mode='pos')
                elif cond_name in ['tibial', 'tib_mixed']:
                    ch_name, latency = envelope.get_peak(tmin=0, tmax=70/1000, mode='pos')

                peak_latency_comp.append(latency)

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
                    if cond_name in ['median', 'med_mixed']:
                        ax[c].set_xlim([0.0, 0.05])
                    elif cond_name in ['tibial', 'tib_mixed']:
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

            # Automatically select the best component, insert 0 if no component passes
            chosen_component = 0
            for sig, lat in sorted(zip(snr_comp, peak_latency_comp), reverse=True):
                if (sig > snr_threshold) and (sep_latency - signal_window <= lat <= sep_latency + signal_window):
                    chosen_component = snr_comp.index(sig) + 1
                    break

            # Insert into the correct place in the excel
            df_comp.at[subject, f'sigma_{cond_name}_comp'] = chosen_component
            if chosen_component == 0:
                df_vis.at[subject, f'Sigma_{cond_name.capitalize()}_Visible'] = 'F'
            else:
                df_vis.at[subject, f'Sigma_{cond_name.capitalize()}_Visible'] = 'T'

        with pd.ExcelWriter(component_fname, mode='a', if_sheet_exists='overlay') as writer:
            df_comp.to_excel(writer, sheet_name=component_sheetname)
        with pd.ExcelWriter(visibility_fname, mode='a', if_sheet_exists='overlay') as writer:
            df_vis.to_excel(writer, sheet_name=visibility_sheetname)

        if save_to_excel:
            # Create data frame
            if cond_name in ['median', 'med_mixed']:
                df_med = pd.DataFrame(snr_cond, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4'])
            elif cond_name in ['tibial', 'tib_mixed']:
                df_tib = pd.DataFrame(snr_cond, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4'])

    if save_to_excel:
        with pd.ExcelWriter(f'{figure_path}ComponentSNR.xlsx') as writer:
            df_med.to_excel(writer, sheet_name='Median Stimulation')
            df_tib.to_excel(writer, sheet_name='Tibial Stimulation')
