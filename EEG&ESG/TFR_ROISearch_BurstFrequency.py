# Script to plot the time-frequency decomposition of the data - we're using raw data before band pass filtering & CCA
# Data is already concatenated raw data and processed for bad trials
# Use the TFR to get characteristics of the burst frequency in our ROI

import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from Common_Functions.check_excel_exist_freq import check_excel_exist_freq
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    data_types = ['Spinal', 'Thalamic', 'Cortical']  # Can be Cortical, Thalamic or Spinal here or both

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, 1)

    subjects = np.arange(1, 37)
    sfreq = 5000
    cond_names = ['median', 'tibial']

    fsearch_low = 400
    fsearch_high = 800
    freqs = np.arange(fsearch_low-50, fsearch_high+50, 3.)
    fmin, fmax = freqs[[0, -1]]

    for data_type in data_types:
        # Make sure our excel sheet is in place to store the values
        excel_fname = f'/data/pt_02718/tmp_data/Peak_Frequency_{fsearch_low}_{fsearch_high}.xlsx'
        sheetname = data_type
        # If fname and sheet exist already - subjects indices will already be in file from initial creation **
        check_excel_exist_freq(subjects, excel_fname, sheetname)
        df_freq = pd.read_excel(excel_fname, sheetname)
        df_freq.set_index('Subject', inplace=True)

        # To use mne grand_average method, need to generate a list of evoked potentials for each subject
        for cond_name in cond_names:  # Conditions (median, tibial)
            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'Tibial - Stimulation'
                time_edge = 0.006
                if data_type == 'Cortical':
                    channel = ['Cz']
                    time_peak = 0.04
                elif data_type == 'Thalamic':
                    channel = ['Cz']
                    time_peak = 0.03
                elif data_type == 'Spinal':
                    channel = ['L1']
                    time_peak = 0.022

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'Median - Stimulation'
                time_edge = 0.003
                if data_type == 'Cortical':
                    channel = ['CP4']
                    time_peak = 0.02
                elif data_type == 'Thalamic':
                    channel = ['CP4']
                    time_peak = 0.014
                elif data_type == 'Spinal':
                    channel = ['SC6']
                    time_peak = 0.013

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                if data_type in ['Cortical', 'Thalamic']:
                    input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
                    fname = f"noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif"
                elif data_type == 'Spinal':
                    input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
                    fname = f"ssp6_cleaned_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Get markers of the boundary so we can have epochs for first 2 blocks, versus second 2 blocks
                time_of_boundaries = []
                # Duration of boundaries is zero, so add a little to the top and we're in the second half of trials
                for jsegment in range(len(raw.annotations)):
                    if raw.annotations.description[jsegment] == 'BAD boundary':  # Find onset time of boundaries
                        time_of_boundaries.append(raw.annotations.onset[jsegment])

                raw_1 = raw.copy().crop(tmin=0.0, tmax=time_of_boundaries[1], include_tmax=False)
                raw_2 = raw.copy().crop(tmin=time_of_boundaries[1], tmax=None, include_tmax=True)

                for col, raw_data in zip(
                        [f'Peak_Frequency_{cond_name}', f'Peak_Frequency_1_{cond_name}', f'Peak_Frequency_2_{cond_name}'],
                        [raw, raw_1, raw_2]):
                    # Evoked Power - Get evoked and then compute power
                    evoked = evoked_from_raw(raw_data, iv_epoch, iv_baseline, trigger_name, False)
                    if data_type in ['Cortical', 'Thalamic']:
                        evoked.reorder_channels(eeg_chans)
                    elif data_type == 'Spinal':
                        evoked.reorder_channels(esg_chans)
                    evoked = evoked.pick_channels(channel)
                    # Want to compute in the same channel for all subjects, ignore if it's marked bad
                    if channel[0] in evoked.info['bads']:
                        evoked.info['bads'] = []
                    power_evoked = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)

                    # Get our ROI and find the peak frequency and associated time of peak
                    power_cropped = power_evoked.crop(tmin=time_peak-time_edge, tmax=time_peak+time_edge, fmin=fsearch_low,
                                                      fmax=fsearch_high, include_tmax=True)
                    roi = np.squeeze(power_cropped.data, 0) # n_freqs, n_times - dropped channel dim as we keep just 1
                    index_of_max = np.unravel_index(np.argmax(roi),roi.shape)

                    # Add values to our dataframe
                    df_freq.at[subject, col] = power_cropped.freqs[index_of_max[0]]
                    if col == f'Peak_Frequency_{cond_name}':
                        df_freq.at[subject, f'Peak_Time_{cond_name}'] = power_cropped.times[index_of_max[1]]

        # Write the dataframe to the excel file
        with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            df_freq.to_excel(writer, sheet_name=sheetname)