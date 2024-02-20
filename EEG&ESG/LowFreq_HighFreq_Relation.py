# Want to extract timing and amplitude of low frequency potentials, versus peak of high frequency amplitude
# envelopes
# Looking at both spinal and cortical peaks
# Mixed nerve condition for both dataset 1 and 2

import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.get_conditioninfo import get_conditioninfo
import pandas as pd
import matplotlib as mpl
from Common_Functions.check_excel_exist_relation import check_excel_exist_relation
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    data_types = ['Spinal', 'Cortical']  # Can be Cortical or Spinal here or both

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    srmr_nr = 2
    sfreq = 5000
    freq_band = 'sigma'

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]

        # Cortical Excel file
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA')
        df_cortical.set_index('Subject', inplace=True)

        # Spinal Excel file
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Updated.xlsx')
        df_spinal = pd.read_excel(xls, 'CCA')
        df_spinal.set_index('Subject', inplace=True)

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]

        # Cortical Excel file
        xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA')
        df_cortical.set_index('Subject', inplace=True)

        # Spinal Excel file
        xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_Updated.xlsx')
        df_spinal = pd.read_excel(xls, 'CCA')
        df_spinal.set_index('Subject', inplace=True)

    for data_type in data_types:
        # Make sure our excel sheet is in place to store the values
        if srmr_nr == 1:
            excel_fname = f'/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx'
        elif srmr_nr == 2:
            excel_fname = f'/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx'
        sheetname = data_type
        # If fname and sheet exist already - subjects indices will already be in file from initial creation **
        check_excel_exist_relation(subjects, excel_fname, sheetname)
        df_rel = pd.read_excel(excel_fname, sheetname)
        df_rel.set_index('Subject', inplace=True)

        # To use mne grand_average method, need to generate a list of evoked potentials for each subject
        for condition in conditions:  # Conditions (median, tibial)
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            if cond_name in ['tibial', 'tib_mixed']:
                time_edge = 0.006
                if data_type == 'Cortical':
                    channel = ['Cz']
                    time_peak = 0.04
                    pot_name = 'P39'
                elif data_type == 'Spinal':
                    pot_name = 'N22'
                    channel = ['L1']
                    time_peak = 0.022

            elif cond_name in ['median', 'med_mixed']:
                time_edge = 0.004
                if data_type == 'Cortical':
                    channel = ['CP4']
                    time_peak = 0.02
                    pot_name = 'N20'
                elif data_type == 'Spinal':
                    channel = ['SC6']
                    time_peak = 0.013
                    pot_name = 'N13'

            for subject in subjects:  # All subjects
                eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                subject_id = f'sub-{str(subject).zfill(3)}'

                if data_type == 'Cortical':
                    if srmr_nr == 1:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                        df = df_cortical

                        # Low Freq SEP
                        input_path_low = "/data/pt_02068/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                        fname_low = f"cnt_clean_{cond_name}.set"
                    elif srmr_nr == 2:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"
                        df = df_cortical

                        # Low Freq SEP
                        input_path_low = "/data/pt_02151/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                        fname_low = f"cnt_clean_{cond_name}.set"

                    raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)
                    evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    evoked_low.crop(tmin=-0.06, tmax=0.07)

                elif data_type == 'Spinal':
                    if srmr_nr == 1:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                        df = df_spinal

                        # Low Freq SEP
                        input_path_low = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                        fname_low = f"epochs_{cond_name}.fif"
                        epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                        evoked_low = epochs_low.average()

                    elif srmr_nr == 2:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data_2/cca/" + subject_id + "/"
                        df = df_spinal

                        # Low Freq SEP
                        input_path_low = f"/data/pt_02569/tmp_data_2/ssp_py/{subject_id}/esg/prepro/6 projections/"
                        fname_low = f"ssp_cleaned_{cond_name}.fif"
                        raw = mne.io.read_raw_fif(input_path_low + fname_low, preload=True)
                        evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        evoked_low.crop(tmin=-0.06, tmax=0.07)

                # Select correct channel for raw ESG data and cca corrected data
                evoked_low = evoked_low.pick_channels(channel)
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if channel_no != 0:
                    channel_cca = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    epochs = epochs.pick_channels([channel_cca])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel_cca)
                    evoked = epochs.copy().average()
                    evoked.crop(tmin=-0.06, tmax=0.07)
                    envelope = evoked.apply_hilbert(envelope=True)

                # Get timing and amplitude of both peaks
                # Look negative for low freq N20, N22, N13, look positive for P39
                # Ampitude envelope always look positive
                # Low Freq
                # First check there is a negative/positive potential to be found
                data_low = evoked_low.crop(tmin=time_peak-time_edge, tmax=time_peak+time_edge).get_data().reshape(-1)
                if data_type == 'Cortical' and cond_name in ['tibial', 'tib_mixed']:
                    if max(data_low) > 0:
                        _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge,
                                                                             tmax=time_peak + time_edge,
                                                                             mode='pos', return_amplitude=True)
                    else:
                        latency_low = np.nan
                        amplitude_low = np.nan
                else:
                    if min(data_low) < 0:
                        _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge,
                                                                            tmax=time_peak + time_edge,
                                                                            mode='neg', return_amplitude=True)
                    else:
                        latency_low = np.nan
                        amplitude_low = np.nan
                # High Freq
                if channel_no != 0:
                    _, latency_high, amplitude_high = envelope.get_peak(tmin=time_peak - time_edge,
                                                                        tmax=time_peak + time_edge,
                                                                        mode='pos', return_amplitude=True)
                else:
                    latency_high = np.nan
                    amplitude_high = np.nan

                df_rel.at[subject, f'{pot_name}'] = latency_low
                df_rel.at[subject, f'{pot_name}_amplitude'] = amplitude_low*10**6
                df_rel.at[subject, f'{pot_name}_high'] = latency_high
                df_rel.at[subject, f'{pot_name}_high_amplitude'] = amplitude_high

        # Write the dataframe to the excel file
        with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            df_rel.to_excel(writer, sheet_name=sheetname)