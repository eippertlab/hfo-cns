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
from Common_Functions.check_excel_exist_general import check_excel_exist_general
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    plot_image = True
    save_to_excel = True  # If we want to save the SNR values on each run

    freq_band = 'sigma'
    data_type = 'subcortical'  # cortical, subcortical or spinal
    srmr_nr = 2
    re_ref = True

    if re_ref == True and data_type == 'spinal' or data_type == 'subcortical':
        raise RuntimeError('Reref cannot be done with data type spinal or subcortical')

    if srmr_nr == 1:
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [2, 3]  # Conditions of interest
        col_names = ['Subject', 'Sigma_Median_Visible', 'Sigma_Tibial_Visible']
        append = ''

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
        conditions = [3, 5]  # Conditions of interest - med_mixed and tib_mixed [3, 5]
        col_names = ['Subject', 'Sigma_Med_mixed_Visible', 'Sigma_Tib_mixed_Visible']
        append = '_2'

    if data_type == 'cortical':
        stem = 'cort'
        if re_ref:
            visibility_fname = f'/data/pt_02718/tmp_data{append}/Visibility_Updated_LF_reref.xlsx'
            figure_path = f'/data/p_02718/Images{append}/EEG_Cort/SNR&EnvelopePeak_reref/'
        else:
            visibility_fname = f'/data/pt_02718/tmp_data{append}/Visibility_Updated_LF.xlsx'
            figure_path = f'/data/p_02718/Images{append}/EEG_Cort/SNR&EnvelopePeak/'
        input_path = f"/data/pt_02718/tmp_data{append}/freq_banded_eeg/"
    elif data_type == 'subcortical':
        stem = 'sub'
        visibility_fname = f'/data/pt_02718/tmp_data{append}/Visibility_Thalamic_Updated_LF.xlsx'
        figure_path = f'/data/p_02718/Images{append}/EEG_Sub/SNR&EnvelopePeak/'
        input_path = f"/data/pt_02718/tmp_data{append}/freq_banded_eeg/"
    else:
        stem = 'spinal'
        visibility_fname = f'/data/pt_02718/tmp_data{append}/Visibility_Spinal_Updated_LF.xlsx'
        figure_path = f'/data/p_02718/Images{append}/ESG/SNR&EnvelopePeak/'
        input_path = f"/data/pt_02718/tmp_data{append}/freq_banded_esg/"
    os.makedirs(figure_path, exist_ok=True)

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    # Check the component file is already generated - want to store the flipping info in the same place so easier to
    # do it this way
    # If fname and sheet exist already - subjects indices will already be in file from initial creation **
    visibility_sheetname = 'LF_Vis'
    check_excel_exist_general(subjects, visibility_fname, visibility_sheetname, col_names)

    df_vis = pd.read_excel(visibility_fname, visibility_sheetname)
    df_vis.set_index('Subject', inplace=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    snr_threshold = 5

    for condition in conditions:
        snr_cond = [0 for x in range(len(subjects))]

        # Set variables
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        for subject in subjects:
            fig, ax = plt.subplots(1, 1)
            subject_id = f'sub-{str(subject).zfill(3)}'
            fname = f"{subject_id}/{freq_band}_{cond_name}.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)

            if data_type == 'cortical' and re_ref is True:
                raw.set_eeg_reference(ref_channels='average')
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline), preload=True)

            if cond_name in ['median', 'med_mixed']:
                sep_latency = df_timing.loc[df_timing['Name'] == f'centre_{stem}_med', 'Time'].iloc[0]/1000
                signal_window = df_timing.loc[df_timing['Name'] == f'edge_{stem}_med', 'Time'].iloc[0]/1000
                if data_type == 'cortical':
                    channel = 'CP4'
                elif data_type == 'subcortical':
                    channel = 'CP4'
                else:
                    channel = 'SC6'
            elif cond_name in ['tibial', 'tib_mixed']:
                sep_latency = df_timing.loc[df_timing['Name'] == f'centre_{stem}_tib', 'Time'].iloc[0]/1000
                signal_window = df_timing.loc[df_timing['Name'] == f'edge_{stem}_tib', 'Time'].iloc[0]/1000
                if data_type == 'cortical':
                    channel = 'Cz'
                elif data_type == 'subcortical':
                    channel = 'Cz'
                else:
                    channel = 'L1'

            snr_comp = []
            peak_latency_comp = []

            # Need to pick channel
            epochs_ch = epochs.copy().pick([channel])

            if cond_name in ['median', 'med_mixed']:
                evoked = epochs_ch.crop(tmin=-0.1, tmax=0.05).copy().average()
            elif cond_name in ['tibial', 'tib_mixed']:
                evoked = epochs_ch.crop(tmin=-0.1, tmax=0.07).copy().average()

            # Get SNR of HFO
            noise_window = [-100/1000, -10/1000]
            snr = calculate_snr(evoked.copy(), noise_window, signal_window, sep_latency, data_type)
            snr_cond[subject-1] = snr

            # Get Envelope
            envelope = evoked.copy().apply_hilbert(envelope=True)
            data = envelope.get_data()
            if cond_name in ['median', 'med_mixed']:
                ch_name, latency = envelope.get_peak(tmin=0, tmax=50/1000, mode='pos')
            elif cond_name in ['tibial', 'tib_mixed']:
                ch_name, latency = envelope.get_peak(tmin=0, tmax=70/1000, mode='pos')

            if plot_image:
                # Plot Envelope with SNR
                ax.plot(evoked.times, evoked.get_data().reshape(-1), color='lightblue')
                ax.plot(evoked.times, data.reshape(-1))
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.axvline(x=sep_latency-signal_window, color='r', linewidth='1')
                ax.axvline(x=sep_latency, color='green', linewidth='1')
                if data_type == 'subcortical':
                    ax.axvline(x=sep_latency+signal_window/2, color='r', linewidth='1')
                else:
                    ax.axvline(x=sep_latency+signal_window, color='r', linewidth='1')
                ax.set_title(f'SNR: {snr:.2f}')
                if cond_name in ['median', 'med_mixed']:
                    ax.set_xlim([0.0, 0.05])
                elif cond_name in ['tibial', 'tib_mixed']:
                    ax.set_xlim([0.0, 0.07])
                # Add lines for mean+-3*std of noise period
                noise_data = evoked.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data().reshape(-1)
                noise_mean = np.mean(noise_data)
                noise_std = np.std(noise_data)
                ax.axhline(y=noise_mean - 3 * noise_std, color='blue', linewidth='1')
                ax.axhline(y=noise_mean + 3 * noise_std, color='blue', linewidth='1')
                plt.suptitle(f'Subject {subject}, {trigger_name}')
                plt.tight_layout()
            plt.savefig(figure_path+f'{subject_id}_{cond_name}')
            plt.close()

            # Automatically select the best component, insert 0 if no component passes
            chosen_component = 0
            if data_type == 'subcortical':
                if (snr > snr_threshold) and (sep_latency - signal_window <= latency <= sep_latency + signal_window/2):
                    chosen_component = 1
            else:
                if (snr > snr_threshold) and (sep_latency - signal_window <= latency <= sep_latency + signal_window):
                    chosen_component = 1

            if chosen_component == 0:
                df_vis.at[subject, f'Sigma_{cond_name.capitalize()}_Visible'] = 'F'
            else:
                df_vis.at[subject, f'Sigma_{cond_name.capitalize()}_Visible'] = 'T'

        with pd.ExcelWriter(visibility_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            df_vis.to_excel(writer, sheet_name=visibility_sheetname)

        if save_to_excel:
            # Create data frame
            if cond_name in ['median', 'med_mixed']:
                df_med = pd.DataFrame(snr_cond, columns=[f'{channel}_snr'])
            elif cond_name in ['tibial', 'tib_mixed']:
                df_tib = pd.DataFrame(snr_cond, columns=[f'{channel}_snr'])

    if save_to_excel:
        with pd.ExcelWriter(f'{figure_path}ComponentSNR.xlsx') as writer:
            df_med.to_excel(writer, sheet_name='Median Stimulation')
            df_tib.to_excel(writer, sheet_name='Tibial Stimulation')
