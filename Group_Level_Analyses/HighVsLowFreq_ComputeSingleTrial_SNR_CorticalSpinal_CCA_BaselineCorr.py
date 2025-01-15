# Get the single trial SNR for each subject and condition
# Looking at only spinal and cortical data
# Tricky: CCA data has some trials excluded based on annotations, need to exclude same trials from low freq epochs

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
    # data_types = ['Spinal']  # Can be Cortical, Spinal here or both
    # data_types = ['Cortical']
    data_types = ['Spinal', 'Cortical']

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    sfreq = 5000
    freq_band = 'sigma'
    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    for srmr_nr in [1, 2]:
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

        # Cortical Excel file low freq
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated_LF.xlsx')
        df_cortical_lf = pd.read_excel(xls, 'CCA')
        df_cortical_lf.set_index('Subject', inplace=True)

        # Spinal Excel file
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
        df_spinal = pd.read_excel(xls, 'CCA')
        df_spinal.set_index('Subject', inplace=True)

        # Spinal Excel file
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated_LF.xlsx')
        df_spinal_lf = pd.read_excel(xls, 'CCA')
        df_spinal_lf.set_index('Subject', inplace=True)

        for data_type in data_types:
            for condition in conditions:  # Conditions (median, tibial)
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name

                for subject in subjects:  # All subjects
                    noise_window = [-100 / 1000, -10 / 1000]
                    epoch_window = [-100 / 1000, 299 / 1000]

                    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    save_path = f'/data/pt_02718/{folder}/singletrial_snr_cca_baselinecorr/{subject_id}/'
                    os.makedirs(save_path, exist_ok=True)

                    if cond_name in ['tibial', 'tib_mixed']:
                        if data_type == 'Cortical':
                            time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_cort_tib', 'Time'].iloc[0])
                            time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_cort_tib', 'Time'].iloc[0])
                            pot_name = 'P39'
                        elif data_type == 'Spinal':
                            pot_name = 'N22'
                            time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_spinal_tib', 'Time'].iloc[0])
                            time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_spinal_tib', 'Time'].iloc[0])

                    elif cond_name in ['median', 'med_mixed']:
                        if data_type == 'Cortical':
                            time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_cort_med', 'Time'].iloc[0])
                            time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_cort_med', 'Time'].iloc[0])
                            pot_name = 'N20'
                        elif data_type == 'Spinal':
                            time_peak = int(df_timing.loc[df_timing['Name'] == 'centre_spinal_med', 'Time'].iloc[0])
                            time_edge = int(df_timing.loc[df_timing['Name'] == 'edge_spinal_med', 'Time'].iloc[0])
                            pot_name = 'N13'

                    # Need in seconds
                    time_peak /= 1000
                    time_edge_neg = time_edge/1000
                    time_edge_pos = time_edge/1000

                    if data_type == 'Cortical':
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = f"/data/pt_02718/{folder}/cca_eeg/{subject_id}/"
                        df = df_cortical

                        # Low Freq SEP
                        input_path_low = f"/data/pt_02718/{folder}/cca_eeg_low/{subject_id}/"
                        fname_low = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"
                        df_low = df_cortical_lf

                        # Set interpolation window (different for eeg and esg data, both in seconds)
                        tstart_interp = -0.0015
                        tend_interp = 0.006

                    elif data_type == 'Spinal':
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = f"/data/pt_02718/{folder}/cca/{subject_id}/"
                        df = df_spinal

                        # Low Freq SEP
                        input_path_low = f"/data/pt_02718/{folder}/cca_low/{subject_id}/"
                        fname_low = f"ssp6_cleaned_{cond_name}.fif"
                        df_low = df_spinal_lf

                        # Set interpolation window (different for eeg and esg data, both in seconds)
                        tstart_interp = -0.007
                        tend_interp = 0.007

                    # Get correct channel for cca HFO data
                    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    if channel_no != 0:
                        channel_cca = f'Cor{channel_no}'
                        inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                        epochs = mne.read_epochs(input_path + fname, preload=True)
                        epochs = epochs.pick([channel_cca])
                        if inv == 'T':
                            epochs.apply_function(invert, picks=channel_cca)

                    # Get correct channel for cca LF data
                    channel_no_low = df_low.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    if channel_no_low != 0:
                        channel_cca = f'Cor{channel_no_low}'
                        inv = df_low.loc[subject, f"{freq_band}_{cond_name}_flip"]
                        epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                        epochs_low = epochs_low.pick([channel_cca])
                        if inv == 'T':
                            epochs_low.apply_function(invert, picks=channel_cca)

                    # Only do the rest if BOTH channel_no are non zero
                    if channel_no != 0 and channel_no_low != 0:
                        # print(f'{subject_id}, {cond_name}, {epochs_low.__len__()}, {epochs.__len__()}')
                        epochs_low.crop(tmin=epoch_window[0], tmax=epoch_window[1])
                        epochs.crop(tmin=epoch_window[0], tmax=epoch_window[1])

                        if epochs_low.__len__() != epochs.__len__():
                            print(epochs_low.__len__())
                            print(epochs.__len__())
                            raise AssertionError('Number of HF and LF trials should be equal')

                        # Get timing and amplitude of both peaks
                        # Look negative for low freq N20, N22, N13, look positive for P39
                        # For HFO polarity is not important
                        # Np.nan arrays - as big as however many trials were kept
                        low_snr = np.full(epochs_low.__len__(), np.nan)
                        high_snr = np.full(epochs_low.__len__(), np.nan)
                        for n in np.arange(0, epochs_low.__len__()):
                            ### Low Freq ###
                            evoked_low = epochs_low[n].average()
                            # P39
                            if data_type == 'Cortical' and cond_name in ['tibial', 'tib_mixed']:
                                _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge_neg,
                                                                                     tmax=time_peak + time_edge_pos,
                                                                                     mode='pos', return_amplitude=True,
                                                                                    strict=False)

                                # Also get the mean value in the epoch window without the interpolation window
                                data = evoked_low.copy().crop(tmin=epoch_window[0], tmax=epoch_window[1]).get_data()
                                indices = evoked_low.time_as_index([tstart_interp, tend_interp], use_rounding=False)
                                data = np.delete(data, np.arange(indices[0], indices[1]), axis=None)
                                mean_base = data.mean()
                                amplitude_diff = amplitude_low - mean_base

                                # Since strict is false, make sure amplitude is positive, otherwise leave as nan
                                if amplitude_low > 0:
                                    data = evoked_low.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data()
                                    sd = data.std()
                                    snr_low = abs(amplitude_diff / sd)
                                else:
                                    snr_low = np.nan
                            # N20, N22, N13
                            else:
                                _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge_neg,
                                                                                    tmax=time_peak + time_edge_pos,
                                                                                    mode='neg', return_amplitude=True,
                                                                                    strict=False)

                                # Also get the mean value in the epoch window without the interpolation window
                                data = evoked_low.copy().crop(tmin=epoch_window[0], tmax=epoch_window[1]).get_data()
                                indices = evoked_low.time_as_index([tstart_interp, tend_interp], use_rounding=False)
                                data = np.delete(data, np.arange(indices[0], indices[1]), axis=None)
                                mean_base = data.mean()
                                amplitude_diff = amplitude_low - mean_base

                                # Since strict is false, make sure amplitude is negative, otherwise leave as nan
                                if amplitude_low < 0:
                                    data = evoked_low.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data()
                                    sd = data.std()
                                    snr_low = abs(amplitude_diff / sd)
                                else:
                                    snr_low = np.nan

                            ### High Freq ###
                            evoked = epochs[n].average()
                            _, latency_high, amplitude_high = evoked.get_peak(tmin=time_peak - time_edge_neg,
                                                                              tmax=time_peak + time_edge_pos,
                                                                              mode='abs', return_amplitude=True)

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