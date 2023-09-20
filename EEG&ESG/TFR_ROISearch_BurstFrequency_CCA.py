# Script to plot the time-frequency decomposition of the data - we're using the CCA processed data
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
from Common_Functions.invert import invert
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    data_types = ['Cortical', 'Spinal']  # Can be Cortical or Spinal here or both
    freq_band = 'sigma'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    # Cortical Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    # Spinal Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, 1)

    subjects = np.arange(1, 37)
    sfreq = 5000
    cond_names = ['median', 'tibial']

    freqs = np.arange(350., 900., 3.)
    fmin, fmax = freqs[[0, -1]]
    fsearch_low = 400
    fsearch_high = 800

    for data_type in data_types:
        # Make sure our excel sheet is in place to store the values
        excel_fname = '/data/pt_02718/tmp_data/Peak_Frequency_CCA.xlsx'
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
                time_edge = 0.004
                if data_type == 'Cortical':
                    time_peak = 0.04
                elif data_type == 'Spinal':
                    time_peak = 0.022

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'Median - Stimulation'
                time_edge = 0.002
                if data_type == 'Cortical':
                    time_peak = 0.02
                elif data_type == 'Spinal':
                    time_peak = 0.013

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                if data_type == 'Cortical':
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                    df = df_cortical
                elif data_type == 'Spinal':
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                    df = df_spinal

                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                channel = f'Cor{channel_no}'
                inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]

                if channel_no != 0:  # 0 marks subjects where no component is selected
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)

                    # Evoked Power - Get evoked and then compute power
                    evoked = epochs.average()

                    # Haven't figured out how/if to do split half reliability since we don't have access to block nos
                    power_evoked = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)

                    # Get our ROI and find the peak frequency and associated time of peak
                    power_cropped = power_evoked.crop(tmin=time_peak-time_edge, tmax=time_peak+time_edge, fmin=fsearch_low,
                                                      fmax=fsearch_high, include_tmax=True)
                    roi = np.squeeze(power_cropped.data, 0) # n_freqs, n_times - dropped channel dim as we keep just 1
                    index_of_max = np.unravel_index(np.argmax(roi),roi.shape)

                    # Add values to our dataframe
                    df_freq.at[subject, f'Peak_Frequency_{cond_name}'] = power_cropped.freqs[index_of_max[0]]
                    df_freq.at[subject, f'Peak_Time_{cond_name}'] = power_cropped.times[index_of_max[1]]

        # Write the dataframe to the excel file
        with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            df_freq.to_excel(writer, sheet_name=sheetname)