# Want to extract timing and amplitude of low frequency potentials, versus peak of high frequency amplitude
# envelopes
# Also includes the same, but for the actual HFO peak (not the envelope)
# Looking at both spinal and cortical peaks
# Mixed nerve condition for both dataset 1 and 2

import mne
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.calculate_snr_hfo import calculate_snr
from Common_Functions.calculate_snr_lowfreq import calculate_snr_lowfreq
from Common_Functions.check_excel_exist_general import check_excel_exist_general
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 2
    sfreq = 5000
    freq_band = 'sigma'

    ##############################################################################################################
    # Set paths and variables
    ##############################################################################################################
    data_types = ['Spinal', 'Cortical']  # Can be Cortical or Spinal here or both

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

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

    # Cortical Excel file - low frequency
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated_LF.xlsx')
    df_cortical_lf = pd.read_excel(xls, 'CCA')
    df_cortical_lf.set_index('Subject', inplace=True)

    # Spinal Excel file
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    # Spinal Excel file - low frequency
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated_LF.xlsx')
    df_spinal_lf = pd.read_excel(xls, 'CCA')
    df_spinal_lf.set_index('Subject', inplace=True)

    ##############################################################################################################
    # Run script to get values and snrs of interest
    ##############################################################################################################
    for data_type in data_types:
        # Make sure our excel sheet is in place to store the values
        excel_fname = f'/data/pt_02718/{folder}/LowFreq_HighFreq_Amp_SNR_CCA.xlsx'
        sheetname = data_type
        if sheetname == 'Cortical':
            col_names = ['Subject', 'N20', 'N20_amplitude', 'N20_high', 'N20_high_amplitude',
                         'N20_SNR', 'N20_high_SNR',
                         'P39', 'P39_amplitude', 'P39_high', 'P39_high_amplitude',
                         'P39_SNR', 'P39_high_SNR']
        elif sheetname == 'Spinal':
            col_names = ['Subject', 'N13', 'N13_amplitude', 'N13_high', 'N13_high_amplitude',
                         'N13_SNR', 'N13_high_SNR',
                         'N22', 'N22_amplitude', 'N22_high', 'N22_high_amplitude',
                         'N22_SNR', 'N22_high_SNR']
        # If fname and sheet exist already - subjects indices will already be in file from initial creation **
        check_excel_exist_general(subjects, excel_fname, sheetname, col_names)
        df_rel = pd.read_excel(excel_fname, sheetname)
        df_rel.set_index('Subject', inplace=True)

        for condition in conditions:  # Conditions (median, tibial) or (med_mixed, tib_mixed)
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'
                noise_window = [-100 / 1000, -10 / 1000]
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
                time_edge /= 1000
                time_peak /= 1000

                ##################################################################################################
                # Select correct files for cortical and spinal data
                #################################################################################################
                if data_type == 'Cortical':
                    # HFO
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = f"/data/pt_02718/{folder}/cca_eeg/{subject_id}/"
                    df = df_cortical

                    # Low Freq SEP
                    input_path_low = f"/data/pt_02718/{folder}/cca_eeg_low/{subject_id}/"
                    fname_low = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"
                    df_low = df_cortical_lf

                elif data_type == 'Spinal':
                    # HFO
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = f"/data/pt_02718/{folder}/cca/{subject_id}/"
                    df = df_spinal

                    # Low Freq SEP
                    input_path_low = f"/data/pt_02718/{folder}/cca_low/{subject_id}/"
                    fname_low = f"ssp6_cleaned_{cond_name}.fif"
                    df_low = df_spinal_lf

                ##################################################################################################
                # Select correct channels for cca low and high freq data
                #################################################################################################
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if channel_no != 0:
                    channel_cca = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    epochs = epochs.pick([channel_cca])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel_cca)
                    evoked = epochs.average()
                    evoked.crop(tmin=-0.1, tmax=0.07)

                channel_no_low = df_low.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if channel_no_low != 0:
                    channel_cca = f'Cor{channel_no_low}'
                    inv = df_low.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                    epochs_low = epochs_low.pick([channel_cca])
                    if inv == 'T':
                        epochs_low.apply_function(invert, picks=channel_cca)
                    evoked_low = epochs_low.average()
                    evoked_low.crop(tmin=-0.1, tmax=0.07)

                # Get timing and amplitude of both peaks
                # Look negative for low freq N20, N22, N13, look positive for P39
                # Get SNR of high and low frequency data
                ####################################################################################################
                # Low Freq
                ####################################################################################################
                # First check there is a negative/positive potential to be found
                if channel_no_low != 0:
                    data_low = evoked_low.copy().crop(tmin=time_peak-time_edge, tmax=time_peak+time_edge).get_data().reshape(-1)
                    if data_type == 'Cortical' and cond_name in ['tibial', 'tib_mixed']:
                        if max(data_low) > 0:
                            _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge,
                                                                                 tmax=time_peak + time_edge,
                                                                                 mode='pos', return_amplitude=True)
                            snr_low = calculate_snr_lowfreq(evoked_low, noise_window, time_edge, time_peak, 'pos')

                        else:
                            latency_low = time_peak
                            amplitude_low = np.nan
                            snr_low = np.nan
                    else:
                        if min(data_low) < 0:
                            _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge,
                                                                                tmax=time_peak + time_edge,
                                                                                mode='neg', return_amplitude=True)
                            snr_low = calculate_snr_lowfreq(evoked_low, noise_window, time_edge, time_peak, 'neg')

                        else:
                            latency_low = time_peak
                            amplitude_low = np.nan
                            snr_low = np.nan

                ####################################################################################################
                # High Freq
                ####################################################################################################
                if channel_no != 0:  # Only do it if there is a component chosen, otherwise insert nans
                    # Look for either polarity for actual HFOs
                    _, latency_high, amplitude_high = evoked.get_peak(tmin=time_peak - time_edge,
                                                                      tmax=time_peak + time_edge,
                                                                      mode='abs', return_amplitude=True)
                    snr_high = calculate_snr(evoked, noise_window, time_edge, time_peak, data_type.lower())
                else:
                    latency_high = time_peak
                    amplitude_high = np.nan
                    snr_high = np.nan

                if channel_no_low != 0:
                    df_rel.at[subject, f'{pot_name}'] = latency_low
                    df_rel.at[subject, f'{pot_name}_amplitude'] = amplitude_low
                    df_rel.at[subject, f'{pot_name}_SNR'] = snr_low

                if channel_no != 0:
                    df_rel.at[subject, f'{pot_name}_high'] = latency_high
                    df_rel.at[subject, f'{pot_name}_high_amplitude'] = amplitude_high
                    df_rel.at[subject, f'{pot_name}_high_SNR'] = snr_high

        # Write the dataframe to the excel file
        with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            df_rel.to_excel(writer, sheet_name=sheetname)