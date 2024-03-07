# Script to plot the time-frequency decomposition of the data - we're using raw data before band pass filtering & CCA
# Data is already concatenated raw data and processed for bad trials
# Use the TFR to get characteristics of the burst frequency in our ROI
# Looks at the mixed nerve conditions in dataset 1 and 2

import mne
import os
import pickle
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from Common_Functions.check_excel_exist_freq import check_excel_exist_freq
from Common_Functions.appy_cca_weights import apply_cca_weights
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    data_types = ['Spinal', 'Thalamic', 'Cortical']  # Can be Cortical, Thalamic or Spinal here or both

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    sfreq = 5000
    fsearch_low = 400
    fsearch_high = 800
    freq_band = 'sigma'
    freqs = np.arange(fsearch_low - 50, fsearch_high + 50, 3.)
    fmin, fmax = freqs[[0, -1]]

    for srmr_nr in [1, 2]:
        if srmr_nr == 1:
            subjects = np.arange(1, 37)
            conditions = [2, 3]

        elif srmr_nr == 2:
            subjects = np.arange(1, 25)
            conditions = [3, 5]

        brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

        for data_type in data_types:
            if srmr_nr == 1:
                folder = 'tmp_data'
            elif srmr_nr == 2:
                folder = 'tmp_data_2'

            # Cortical Excel files
            xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated.xlsx')
            df_cortical = pd.read_excel(xls, 'CCA')
            df_cortical.set_index('Subject', inplace=True)

            # Thalamic Excel files
            xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Thalamic_Updated.xlsx')
            df_thalamic = pd.read_excel(xls, 'CCA')
            df_thalamic.set_index('Subject', inplace=True)

            # Spinal Excel files
            xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
            df_spinal = pd.read_excel(xls, 'CCA')
            df_spinal.set_index('Subject', inplace=True)

            # Make sure our excel sheet is in place to store the values, and get the correct weights we need to apply
            excel_fname = f'/data/pt_02718/{folder}/Peak_Frequency_{fsearch_low}_{fsearch_high}_ccabroadband.xlsx'

            sheetname = data_type
            # If fname and sheet exist already - subjects indices will already be in file from initial creation **
            check_excel_exist_freq(subjects, excel_fname, sheetname, srmr_nr)
            df_freq = pd.read_excel(excel_fname, sheetname)
            df_freq.set_index('Subject', inplace=True)

            # To use mne grand_average method, need to generate a list of evoked potentials for each subject
            for condition in conditions:
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                if cond_name in ['tibial', 'tib_mixed']:
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

                elif cond_name in ['median', 'med_mixed']:
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
                    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

                    if data_type == 'Spinal':
                        df = df_spinal
                        with open(f'/data/pt_02718/{folder}/cca/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                                  'rb') as f:
                            W_st = pickle.load(f)
                    elif data_type == 'Thalamic':
                        df = df_thalamic
                        with open(f'/data/pt_02718/{folder}/cca_eeg_thalamic/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                                  'rb') as f:
                            W_st = pickle.load(f)
                    elif data_type == 'Cortical':
                        df = df_cortical
                        with open(f'/data/pt_02718/{folder}/cca_eeg/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                                  'rb') as f:
                            W_st = pickle.load(f)
                    else:
                        raise RuntimeError('This given datatype is not one of Spinal/Thalamic/Cortical')

                    if data_type in ['Cortical', 'Thalamic']:
                        fname = f"noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif"
                        input_path = f"/data/pt_02718/{folder}/imported/{subject_id}/"

                    elif data_type == 'Spinal':
                        fname = f"ssp6_cleaned_{cond_name}.fif"
                        input_path = f"/data/pt_02718/{folder}/ssp_cleaned/{subject_id}/"

                    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                    if srmr_nr == 1:
                        # Get markers of the boundary so we can have epochs for first 2 blocks, versus second 2 blocks
                        time_of_boundaries = []
                        # Duration of boundaries is zero, so add a little to the top and we're in the second half of trials
                        for jsegment in range(len(raw.annotations)):
                            if raw.annotations.description[jsegment] == 'BAD boundary':  # Find onset time of boundaries
                                time_of_boundaries.append(raw.annotations.onset[jsegment])

                        raw_1 = raw.copy().crop(tmin=0.0, tmax=time_of_boundaries[1], include_tmax=False)
                        raw_2 = raw.copy().crop(tmin=time_of_boundaries[1], tmax=None, include_tmax=True)

                    # Experiment 2 was recorded without block setup - all 2000 trials just done together
                    # Not a true split half this way - may not use results in full analysis
                    elif srmr_nr == 2:
                        raw_1 = raw.copy().crop(tmin=0.0, tmax=raw.times[int(np.floor(raw.n_times/2))], include_tmax=False)
                        raw_2 = raw.copy().crop(tmin=raw.times[int(np.floor(raw.n_times/2))+1], tmax=None, include_tmax=True)

                    for col, raw_data in zip(
                            [f'Peak_Frequency_{cond_name}', f'Peak_Frequency_1_{cond_name}', f'Peak_Frequency_2_{cond_name}'],
                            [raw, raw_1, raw_2]):
                        # Evoked Power - Get evoked and then compute power
                        # Need to change this to get epochs, apply weights to epochs, then calculate evoked
                        events, event_ids = mne.events_from_annotations(raw_data)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw_data, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline), preload=True)

                        apply_weights_kwargs = dict(
                            weights = W_st
                        )
                        cor_names = [f'Cor{i}' for i in np.arange(1, len(W_st) + 1)]
                        if data_type == 'Spinal':
                            if cond_name in ['median', 'med_mixed']:
                                epochs.pick(cervical_chans).reorder_channels(cervical_chans)
                                epochs_ccafiltered = epochs.apply_function(apply_cca_weights, channel_wise=False, **apply_weights_kwargs)
                                # Remap names to match Cor 1, Cor2 etc
                                channel_map = {cervical_chans[i]: cor_names[i] for i in range(len(cervical_chans))}
                                epochs_ccafiltered.rename_channels(channel_map)
                            elif cond_name in ['tibial', 'tib_mixed']:
                                epochs.pick(lumbar_chans).reorder_channels(lumbar_chans)
                                epochs_ccafiltered = epochs.apply_function(apply_cca_weights, channel_wise=False, **apply_weights_kwargs)
                                channel_map = {lumbar_chans[i]: cor_names[i] for i in range(len(lumbar_chans))}
                                epochs_ccafiltered.rename_channels(channel_map)
                        elif data_type in ['Thalamic', 'Cortical']:
                            epochs.pick(eeg_chans).reorder_channels(eeg_chans)
                            epochs_ccafiltered = epochs.apply_function(apply_cca_weights, picks=eeg_chans, channel_wise=False, **apply_weights_kwargs)
                            channel_map = {eeg_chans[i]: cor_names[i] for i in range(len(eeg_chans))}
                            epochs_ccafiltered.rename_channels(channel_map)

                        #  evoked = evoked_from_raw(raw_data, iv_epoch, iv_baseline, trigger_name, False)
                        evoked = epochs_ccafiltered.average()
                        channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                        channel = f'Cor{channel_no}'
                        inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]

                        if channel_no != 0:  # 0 marks subjects where no component is selected
                            evoked.pick(channel)
                            # Get the correct component from the excel files
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
                        else:
                            # Add values to our dataframe
                            df_freq.at[subject, col] = np.nan
                            if col == f'Peak_Frequency_{cond_name}':
                                df_freq.at[subject, f'Peak_Time_{cond_name}'] = np.nan

            # Write the dataframe to the excel file
            with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
                df_freq.to_excel(writer, sheet_name=sheetname)