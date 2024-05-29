# Get the single trial SNR for each subject and condition
# Looking at only spinal and cortical data
# Tricky: CCA data has some trials excluded based on annotations, need to exclude same trials from low freq epochs
# FInd time of triggers that don't match?

import mne
import os
import numpy as np
import pickle
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.get_conditioninfo import get_conditioninfo
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    data_types = ['Cortical', 'Spinal']  # Can be Cortical, Spinal here or all
    # Difficulties with LF-SEP timings in Thalamic - leave out

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    srmr_nr = 1
    sfreq = 5000
    freq_band = 'sigma'
    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'

    # Cortical Excel file
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    # Spinal Excel file
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    for data_type in data_types:
        for condition in conditions:  # Conditions (median, tibial)
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            for subject in subjects:  # All subjects
                # Np.nan arrays - will stay nan for bottom indices equal to however many trials were dropped
                low_snr = np.full(2000, np.nan)
                high_snr = np.full(2000, np.nan)
                noise_window = [-100 / 1000, -10 / 1000]

                eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                subject_id = f'sub-{str(subject).zfill(3)}'

                save_path = f'/data/pt_02718/{folder}/singletrial_snr/{subject_id}/'
                os.makedirs(save_path, exist_ok=True)

                if cond_name in ['tibial', 'tib_mixed']:
                    if data_type == 'Cortical':
                        channel = ['Cz']
                        time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_cort_tib', 'Time'].iloc[0])
                        time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_cort_tib', 'Time'].iloc[0])
                        pot_name = 'P39'
                    elif data_type == 'Spinal':
                        pot_name = 'N22'
                        channel = ['L1']
                        time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_spinal_tib', 'Time'].iloc[0])
                        time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_spinal_tib', 'Time'].iloc[0])

                elif cond_name in ['median', 'med_mixed']:
                    if data_type == 'Cortical':
                        channel = ['CP4']
                        time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_cort_med', 'Time'].iloc[0])
                        time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_cort_med', 'Time'].iloc[0])
                        pot_name = 'N20'
                    elif data_type == 'Spinal':
                        channel = ['SC6']
                        time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_spinal_med', 'Time'].iloc[0])
                        time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_spinal_med', 'Time'].iloc[0])
                        pot_name = 'N13'

                # Need in seconds
                time_peak /= 1000
                time_edge_neg = time_edge/1000
                time_edge_pos = time_edge/1000

                if data_type == 'Cortical':
                    if srmr_nr == 1:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                        df = df_cortical

                        # Low Freq SEP
                        input_path_low = "/data/pt_02068/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                        fname_low = f"cnt_clean_{cond_name}.set"
                        # Find bad trials
                        input_path_bad = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
                        fname_bad = f"{freq_band}_{cond_name}.fif"

                    elif srmr_nr == 2:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"
                        df = df_cortical

                        # Low Freq SEP
                        input_path_low = "/data/pt_02151/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                        fname_low = f"cnt_clean_{cond_name}.set"
                        # Find bad trials
                        input_path_bad = "/data/pt_02718/tmp_data_2/freq_banded_eeg/" + subject_id + "/"
                        fname_bad = f"{freq_band}_{cond_name}.fif"

                    raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs_low = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                            baseline=tuple(iv_baseline), preload=True)

                    raw_withbadmarked = mne.io.read_raw_fif(input_path_bad + fname_bad, preload=True)
                    events_bad, event_ids_bad = mne.events_from_annotations(raw_withbadmarked)
                    event_id_dict_bad = {key: value for key, value in event_ids_bad.items() if key == trigger_name}
                    epochs_withbads = mne.Epochs(raw_withbadmarked, events_bad, event_id=event_id_dict_bad,
                                                 tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                                 baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)

                elif data_type == 'Spinal':
                    if srmr_nr == 1:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                        df = df_spinal

                        # Low Freq SEP
                        input_path_low = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                        fname_low = f"ssp_cleaned_{cond_name}.fif"
                        # Find bad trials
                        input_path_bad = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
                        fname_bad = f"{freq_band}_{cond_name}.fif"

                    elif srmr_nr == 2:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data_2/cca/" + subject_id + "/"
                        df = df_spinal

                        # Low Freq SEP
                        input_path_low = f"/data/pt_02569/tmp_data_2/ssp_py/{subject_id}/esg/prepro/6 projections/"
                        fname_low = f"ssp_cleaned_{cond_name}.fif"
                        # Find bad trials
                        input_path_bad = "/data/pt_02718/tmp_data_2/freq_banded_esg/" + subject_id + "/"
                        fname_bad = f"{freq_band}_{cond_name}.fif"

                    raw = mne.io.read_raw_fif(input_path_low + fname_low, preload=True)
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs_low = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                        baseline=tuple(iv_baseline), preload=True)

                    raw_withbadmarked = mne.io.read_raw_fif(input_path_bad + fname_bad, preload=True)
                    events_bad, event_ids_bad = mne.events_from_annotations(raw_withbadmarked)
                    event_id_dict_bad = {key: value for key, value in event_ids_bad.items() if key == trigger_name}
                    epochs_withbads = mne.Epochs(raw_withbadmarked, events_bad, event_id=event_id_dict_bad, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                        baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)

                # Select correct channel for raw ESG data
                epochs_low = epochs_low.pick_channels(channel)

                # Get correct channel for cca HFO data
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if channel_no != 0:
                    channel_cca = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    epochs = epochs.pick_channels([channel_cca])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel_cca)

                # Locate bad trials so we can drop the same ones from the low freq data as have been dropped from the
                # HF data before we do CCA
                bad_trials = [ix for ix, log in enumerate(epochs_withbads.drop_log) if log == ('BAD_amp',)]
                # print(bad_trials)
                # print(epochs_low.drop(bad_trials, reason='BAD_amp').__len__())
                # print(epochs.__len__())
                # print(epochs_withbads.__len__())

                # Crop
                epochs_low.crop(tmin=-0.1, tmax=0.07)
                epochs.crop(tmin=-0.1, tmax=0.07)

                # Get timing and amplitude of both peaks
                # Look negative for low freq N20, N22, N13, look positive for P39
                # For HFO polarity is not important
                for n in np.arange(0, epochs.__len__()):
                    # Low Freq
                    evoked_low = epochs_low[n].average()
                    # First check there is a negative/positive potential to be found
                    data_low = evoked_low.copy().crop(tmin=time_peak-time_edge_neg, tmax=time_peak+time_edge_pos).get_data().reshape(-1)
                    if data_type == 'Cortical' and cond_name in ['tibial', 'tib_mixed']:
                        if max(data_low) > 0:
                            _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge_neg,
                                                                                 tmax=time_peak + time_edge_pos,
                                                                                 mode='pos', return_amplitude=True)
                        else:
                            latency_low = time_peak
                            amplitude_low = np.nan
                    else:
                        if min(data_low) < 0:
                            _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge_neg,
                                                                                tmax=time_peak + time_edge_pos,
                                                                                mode='neg', return_amplitude=True)
                        else:
                            latency_low = time_peak
                            amplitude_low = np.nan

                    # High Freq
                    evoked = epochs[n].average()
                    if channel_no != 0:
                        _, latency_high, amplitude_high = evoked.get_peak(tmin=time_peak - time_edge_neg,
                                                                          tmax=time_peak + time_edge_pos,
                                                                          mode='abs', return_amplitude=True)
                    else:
                        latency_high = np.nan
                        amplitude_high = np.nan

                    # Low freq SNR
                    data = evoked_low.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data()
                    sd = data.std()
                    snr_low = abs(amplitude_low / sd)

                    # High freq SNR
                    data = evoked.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data()
                    sd = data.std()
                    snr_high = abs(amplitude_high / sd)

                    low_snr[n] = snr_low
                    high_snr[n] = snr_high

                afile = open(save_path + f'snr_high_{freq_band}_{cond_name}_{data_type.lower()}.pkl', 'wb')
                pickle.dump(high_snr, afile)
                afile.close()

                afile = open(save_path + f'snr_low_{freq_band}_{cond_name}_{data_type.lower()}.pkl', 'wb')
                pickle.dump(low_snr, afile)
                afile.close()