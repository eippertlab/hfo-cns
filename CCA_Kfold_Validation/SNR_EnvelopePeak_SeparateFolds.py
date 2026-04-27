# Calculate SNR of first 4 components in each fold, and check whether peak of envelope is in required zone


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.calculate_snr_hfo import calculate_snr
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


if __name__ == '__main__':

    freq_band = 'sigma'
    srmr_nr = 2
    mode = 'Spinal'  # Can be Brain, Thalamic or Spinal
    kfolds = 5
    n_components = 4

    if srmr_nr == 1:
        app_folder = ""
        median_cols = [f"sigma_median_fold{x + 1}_comp{y + 1}_{option}" for y in range(n_components) for x in range(kfolds) for
                       option in ['SNR', 'Peak']]
        tibial_cols = [f"sigma_tibial_fold{x + 1}_comp{y + 1}_{option}" for y in range(n_components) for x in range(kfolds) for
                       option in ['SNR', 'Peak']]

        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [2, 3]  # Conditions of interest

    elif srmr_nr == 2:
        app_folder = "_2"
        subjects = np.arange(1, 25)  # 1 through 36 to access subject data
        conditions = [3, 5]  # Conditions of interest
        median_cols = [f"sigma_med_mixed_fold{x + 1}_comp{y + 1}_{option}" for y in range(n_components) for x in
                       range(kfolds)
                       for
                       option in ['SNR', 'Peak']]
        tibial_cols = [f"sigma_tib_mixed_fold{x + 1}_comp{y + 1}_{option}" for y in range(n_components) for x in
                       range(kfolds)
                       for
                       option in ['SNR', 'Peak']]

    if mode == 'Brain':
        excel_fname = f'/data/pt_02718/tmp_data{app_folder}/SNR_Peak_{kfolds}fold_EEG_Updated.xlsx'
        excel_sheetname = 'SNR_Peak'
    elif mode == 'Thalamic':
        excel_fname = f'/data/pt_02718/tmp_data{app_folder}/SNR_Peak_{kfolds}fold_EEG_Thalamic_Updated.xlsx'
        excel_sheetname = 'SNR_Peak'
    elif mode == 'Spinal':
        excel_fname = f'/data/pt_02718/tmp_data{app_folder}/SNR_Peak_{kfolds}fold_Updated.xlsx'
        excel_sheetname = 'SNR_Peak'
    else:
        raise ValueError('Mode must be selected')

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]


    df = pd.DataFrame(columns=median_cols+tibial_cols)

    for condition in conditions:
        for subject in subjects:
            for fold in range(kfolds):
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name

                subject_id = f'sub-{str(subject).zfill(3)}'
                fname = f"{fold}fold_{freq_band}_{cond_name}.fif"
                if mode == 'Brain':
                    input_path = f"/data/pt_02718/tmp_data{app_folder}/cca_{kfolds}fold_eeg/" + subject_id + "/"
                    data_type = 'cortical'
                    if cond_name in ['median', 'med_mixed']:
                        sep_latency = df_timing.loc[df_timing['Name'] == 'centre_cort_med', 'Time'].iloc[0] / 1000
                        signal_window = df_timing.loc[df_timing['Name'] == 'edge_cort_med', 'Time'].iloc[0] / 1000
                    elif cond_name in ['tibial', 'tib_mixed']:
                        sep_latency = df_timing.loc[df_timing['Name'] == 'centre_cort_tib', 'Time'].iloc[0] / 1000
                        signal_window = df_timing.loc[df_timing['Name'] == 'edge_cort_tib', 'Time'].iloc[0] / 1000
                elif mode == 'Thalamic':
                    input_path = f"/data/pt_02718/tmp_data{app_folder}/cca_{kfolds}fold_eeg_thalamic/" + subject_id + "/"
                    data_type = 'subcortical'
                    if cond_name in ['median', 'med_mixed']:
                        sep_latency = df_timing.loc[df_timing['Name'] == 'centre_sub_med', 'Time'].iloc[0] / 1000
                        signal_window = df_timing.loc[df_timing['Name'] == 'edge_sub_med', 'Time'].iloc[0] / 1000
                    elif cond_name in ['tibial', 'tib_mixed']:
                        sep_latency = df_timing.loc[df_timing['Name'] == 'centre_sub_tib', 'Time'].iloc[0] / 1000
                        signal_window = df_timing.loc[df_timing['Name'] == 'edge_sub_tib', 'Time'].iloc[0] / 1000
                elif mode == 'Spinal':
                    input_path = f"/data/pt_02718/tmp_data{app_folder}/cca_{kfolds}fold/" + subject_id + "/"
                    data_type = 'spinal'
                    if cond_name in ['median', 'med_mixed']:
                        sep_latency = df_timing.loc[df_timing['Name'] == 'centre_spinal_med', 'Time'].iloc[0] / 1000
                        signal_window = df_timing.loc[df_timing['Name'] == 'edge_spinal_med', 'Time'].iloc[0] / 1000
                    elif cond_name in ['tibial', 'tib_mixed']:
                        sep_latency = df_timing.loc[df_timing['Name'] == 'centre_spinal_tib', 'Time'].iloc[0] / 1000
                        signal_window = df_timing.loc[df_timing['Name'] == 'edge_spinal_tib', 'Time'].iloc[0] / 1000

                epochs = mne.read_epochs(input_path + fname, preload=True)

                for c in np.arange(0, n_components):  # Loop through all components
                    # Need to pick channel
                    channel = f'Cor{c+1}'
                    epochs_ch = epochs.copy().pick_channels([channel])

                    if cond_name in ['median', 'med_mixed']:
                        evoked = epochs_ch.crop(tmin=-0.1, tmax=0.05).copy().average()
                    elif cond_name in ['tibial', 'tib_mixed']:
                        evoked = epochs_ch.crop(tmin=-0.1, tmax=0.07).copy().average()

                    # Get SNR of HFO
                    noise_window = [-100/1000, -10/1000]
                    snr = calculate_snr(evoked.copy(), noise_window, signal_window, sep_latency, data_type)
                    df.at[subject, f"sigma_{cond_name}_fold{fold + 1}_comp{c + 1}_SNR"] = snr

                    # Get Envelope and check if it is within bounds
                    envelope = evoked.copy().apply_hilbert(envelope=True)
                    data = envelope.get_data()
                    if cond_name in ['median', 'med_mixed']:
                        ch_name, latency = envelope.get_peak(tmin=0, tmax=50/1000, mode='pos')
                    elif cond_name in ['tibial', 'tib_mixed']:
                        ch_name, latency = envelope.get_peak(tmin=0, tmax=70/1000, mode='pos')

                    if sep_latency - signal_window <= latency <= sep_latency + signal_window:
                        df.at[subject, f"sigma_{cond_name}_fold{fold + 1}_comp{c + 1}_Peak"] = 'T'
                    else:
                        df.at[subject, f"sigma_{cond_name}_fold{fold + 1}_comp{c + 1}_Peak"] = 'F'

    with pd.ExcelWriter(excel_fname, mode='w', engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=excel_sheetname, columns=median_cols+tibial_cols)

