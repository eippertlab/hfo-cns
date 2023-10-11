# Looking at the average power in a region of interest (400-800Hz), comparing this between the 'correct' and 'incorrect'
# patch in the spinal cord after median/tibial nerve stimulation

# Script to plot the time-frequency decomposition about the spinal triggers for the correct versus incorrect patch
# Also include the cortical data for just the correct cluster
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.check_excel_exist_power import check_excel_exist_power
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 2
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]


    fsearch_low = 400
    fsearch_high = 800
    freqs = np.arange(fsearch_low - 50, fsearch_high + 50, 3.)
    fmin, fmax = freqs[[0, -1]]

    if srmr_nr == 1:
        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
        df_timing_spinal = pd.read_excel(xls_timing, 'Spinal')
        df_timing_spinal.set_index('Subject', inplace=True)

        excel_fname = f'/data/pt_02718/tmp_data/PowerInROI_{fsearch_low}_{fsearch_high}.xlsx'

        xls_burstfreq = f'/data/pt_02718/tmp_data/Peak_Frequency_400_800.xlsx'
        df_freq_spinal = pd.read_excel(xls_burstfreq, 'Spinal')
        df_freq_spinal.set_index('Subject', inplace=True)

        subjects = np.arange(1, 37)
        sfreq = 5000
        conditions = [2, 3]

    elif srmr_nr == 2:
        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx')
        df_timing_spinal = pd.read_excel(xls_timing, 'Spinal')
        df_timing_spinal.set_index('Subject', inplace=True)

        excel_fname = f'/data/pt_02718/tmp_data_2/PowerInROI_{fsearch_low}_{fsearch_high}.xlsx'

        xls_burstfreq = f'/data/pt_02718/tmp_data_2/Peak_Frequency_400_800.xlsx'
        df_freq_spinal = pd.read_excel(xls_burstfreq, 'Spinal')
        df_freq_spinal.set_index('Subject', inplace=True)

        subjects = np.arange(1, 25)
        sfreq = 5000
        conditions = [3, 5]

    sheetname = 'Average Power'
    # If fname and sheet exist already - subjects indices will already be in file from initial creation **
    check_excel_exist_power(subjects, excel_fname, sheetname, srmr_nr)
    df_pow = pd.read_excel(excel_fname, sheetname)
    df_pow.set_index('Subject', inplace=True)
    for condition in conditions:
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'
            burst_freq = round(df_freq_spinal.loc[subject, f"Peak_Frequency_{cond_name}"], 4)

            if cond_name in ['tibial', 'tib_mixed']:
                correct_channel = ['L1']
                incorrect_channel = ['SC6']
                time_peak = 0.022
                time_edge = 0.006

            elif cond_name in ['median', 'med_mixed']:
                correct_channel = ['SC6']
                incorrect_channel = ['L1']
                time_peak = 0.013
                time_edge = 0.003

            # Read in Spinal Data
            if srmr_nr == 1:
                input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
            elif srmr_nr == 2:
                input_path = "/data/pt_02718/tmp_data_2/ssp_cleaned/" + subject_id + "/"
            fname = f"ssp6_cleaned_{cond_name}.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            evoked_spinal = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked_correct = evoked_spinal.copy().pick_channels(correct_channel)
            evoked_incorrect = evoked_spinal.copy().pick_channels(incorrect_channel)

            evoked_correct.crop(tmin=-0.06, tmax=0.1)
            evoked_incorrect.crop(tmin=-0.06, tmax=0.1)

            # Get power
            power_correct = mne.time_frequency.tfr_stockwell(evoked_correct, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)
            power_incorrect = mne.time_frequency.tfr_stockwell(evoked_incorrect, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)

            # Get our ROI and find the peak frequency and associated time of peak
            # ROI: 400-800Hz, xms about peak
            power_correct_cropped = power_correct.crop(tmin=time_peak - time_edge, tmax=time_peak + time_edge,
                                                      fmin=fsearch_low,
                                                      fmax=fsearch_high, include_tmax=True)
            power_incorrect_cropped = power_incorrect.crop(tmin=time_peak - time_edge, tmax=time_peak + time_edge,
                                                       fmin=fsearch_low,
                                                       fmax=fsearch_high, include_tmax=True)
            # ROI: 60Hz about burst freq, xms about peak
            # power_correct_cropped = power_correct.crop(tmin=time_peak - time_edge, tmax=time_peak + time_edge,
            #                                            fmin=burst_freq-30,
            #                                            fmax=burst_freq+30, include_tmax=True)
            # power_incorrect_cropped = power_incorrect.crop(tmin=time_peak - time_edge, tmax=time_peak + time_edge,
            #                                                fmin=burst_freq-30,
            #                                                fmax=burst_freq+30, include_tmax=True)

            roi_correct = np.squeeze(power_correct_cropped.data, 0)  # n_freqs, n_times - dropped channel dim as we keep just 1
            roi_incorrect = np.squeeze(power_incorrect_cropped.data,
                                     0)  # n_freqs, n_times - dropped channel dim as we keep just 1

            # average_correct = roi_correct.mean()
            # average_incorrect = roi_incorrect.mean()

            # Power AT burst frequency
            index_of_max = np.unravel_index(np.argmax(roi_correct), roi_correct.shape)
            average_correct = roi_correct[index_of_max[0], index_of_max[1]]
            average_incorrect = roi_incorrect[index_of_max[0], index_of_max[1]]

            # Add values to our dataframe
            df_pow.at[subject, f'Power_{cond_name}_correct'] = average_correct
            df_pow.at[subject, f'Power_{cond_name}_incorrect'] = average_incorrect
            df_pow.at[subject, f'Power_{cond_name}_ratio'] = average_correct/average_incorrect

    # Write the dataframe to the excel file
    with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
        df_pow.to_excel(writer, sheet_name=sheetname)