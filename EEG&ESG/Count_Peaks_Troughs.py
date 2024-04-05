# Count the total peaks and troughs in the HFO signals at the level of the periphery (no CCA), spinal, subcortical and
# cortical (after CCA) signals
# Get this total within a specific time window on either side of the expected latency
# Test different amplitude thresholds: 10%, 20%, 25%, 50% of max

import numpy as np
import os
import mne
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from Common_Functions.check_excel_exist_general import check_excel_exist_general
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.invert import invert
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


if __name__ == '__main__':
    freq_band = 'sigma'

    for srmr_nr in [1, 2]:
        if srmr_nr == 1:
            subjects = np.arange(1, 37)
            conditions = [2, 3]
            base = ''
            col_names = ['Subject',
                         'bipolar_troughs_median', 'bipolar_peaks_median', 'spinal_troughs_median', 'spinal_peaks_median',
                         'subcortical_troughs_tibial', 'subcortical_peaks_tibial', 'cortical_troughs_tibial',
                         'cortical_peaks_tibial']
        elif srmr_nr == 2:
            subjects = np.arange(1, 25)
            conditions = [3, 5]
            base = '_2'
            col_names = ['Subject',
                         'bipolar_troughs_med_mixed', 'bipolar_peaks_med_mixed', 'spinal_troughs_med_mixed', 'spinal_peaks_med_mixed',
                         'subcortical_troughs_tib_mixed', 'subcortical_peaks_tib_mixed', 'cortical_troughs_tib_mixed',
                         'cortical_peaks_tib_mixed']

        sampling_rate = 10000

        cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
        df = pd.read_excel(cfg_path)
        notch_freq = df.loc[df['var_name'] == 'notch_freq', 'var_value'].iloc[0]

        iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                       df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
        iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                    df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

        # Set up the dataframe to store results
        save_path = f"/data/pt_02718/tmp_data{base}/"

        excel_fname = f'{save_path}Peaks_Troughs.xlsx'

        for excel_sheetname, divisor in zip(['10%', '20%', '25%', '33%', '50%'], [10, 5, 4, 3, 2]):
            check_excel_exist_general(subjects, fname=excel_fname, sheetname=excel_sheetname, col_names=col_names)
            df_pt = pd.read_excel(excel_fname, excel_sheetname)
            df_pt.set_index('Subject', inplace=True)

            for data_type in ['bipolar', 'spinal', 'subcortical', 'cortical']:
                figure_path = f'/data/p_02718/Images{base}/Peak_Trough_Images/{data_type}/'
                os.makedirs(figure_path, exist_ok=True)

                for condition in conditions:
                    cond_info = get_conditioninfo(condition, srmr_nr)
                    cond_name = cond_info.cond_name
                    trigger_name = cond_info.trigger_name
                    stimulation = condition - 1

                    for subject in subjects:
                        flag = False
                        subject_id = f'sub-{str(subject).zfill(3)}'
                        time_edge = 5/1000
                        if data_type == 'bipolar':
                            input_path = f"/data/pt_02718/tmp_data{base}/freq_banded_bipolar/{subject_id}/"
                            fname = f'sigma_{cond_name}.fif'
                            if cond_name in ['median', 'med_mixed']:
                                channel = ['Biceps']
                                ref_lat = 6 / 1000
                            elif cond_name in ['tibial', 'tib_mixed']:
                                channel = ['KneeM']
                                ref_lat = 9 / 1000

                        elif data_type == 'spinal':
                            input_path = f"/data/pt_02718/tmp_data{base}/cca/{subject_id}/"
                            fname = f'sigma_{cond_name}.fif'
                            if cond_name in ['median', 'med_mixed']:
                                ref_lat = 13 / 1000
                            elif cond_name in ['tibial', 'tib_mixed']:
                                ref_lat = 22 / 1000
                            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{base}/Components_Updated.xlsx')
                            df_comp = pd.read_excel(xls, 'CCA')
                            df_comp.set_index('Subject', inplace=True)

                        elif data_type == 'subcortical':
                            input_path = f"/data/pt_02718/tmp_data{base}/cca_eeg_thalamic/{subject_id}/"
                            fname = f'sigma_{cond_name}.fif'
                            if cond_name in ['median', 'med_mixed']:
                                ref_lat = 14 / 1000
                            elif cond_name in ['tibial', 'tib_mixed']:
                                ref_lat = 30 / 1000
                            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{base}/Components_EEG_Thalamic_Updated.xlsx')
                            df_comp = pd.read_excel(xls, 'CCA')
                            df_comp.set_index('Subject', inplace=True)

                        elif data_type == 'cortical':
                            input_path = f"/data/pt_02718/tmp_data{base}/cca_eeg/{subject_id}/"
                            fname = f'sigma_{cond_name}.fif'
                            if cond_name in ['median', 'med_mixed']:
                                ref_lat = 20 / 1000
                            elif cond_name in ['tibial', 'tib_mixed']:
                                ref_lat = 40 / 1000
                            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{base}/Components_EEG_Updated.xlsx')
                            df_comp = pd.read_excel(xls, 'CCA')
                            df_comp.set_index('Subject', inplace=True)

                        if data_type == 'bipolar':
                            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                            events, event_ids = mne.events_from_annotations(raw)
                            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                                baseline=tuple(iv_baseline))
                            evoked = epochs.average(picks='all')
                            evoked_ch = evoked.pick(channel).crop(ref_lat-time_edge, ref_lat+time_edge)

                        else:
                            channel_no = df_comp.loc[subject, f"{freq_band}_{cond_name}_comp"]
                            channel = f'Cor{channel_no}'
                            inv = df_comp.loc[subject, f"{freq_band}_{cond_name}_flip"]
                            if channel_no != 0:  # 0 marks subjects where no component is selected
                                epochs = mne.read_epochs(input_path + fname, preload=True)
                                epochs = epochs.pick_channels([channel])
                                if inv == 'T':
                                    epochs.apply_function(invert, picks=channel)
                                evoked_ch = epochs.average().crop(ref_lat-time_edge, ref_lat+time_edge)
                            else:
                                flag = True  # Won't write to dataframe if the flag is True, since theres no real component
                                epochs = mne.read_epochs(input_path + fname, preload=True)
                                epochs = epochs.pick_channels(['Cor5'])
                                if inv == 'T':
                                    epochs.apply_function(invert, picks=channel)
                                evoked_ch = epochs.average().crop(ref_lat-time_edge, ref_lat+time_edge)

                        data_ch = evoked_ch.get_data().reshape(-1)

                        # Get pos and neg peaks
                        if data_type == 'bipolar':
                            dist = (1/800)*10000
                        else:
                            dist = (1/2100)*10000
                        peaks_pos, props_pos = scipy.signal.find_peaks(data_ch, distance=dist, height=np.max(data_ch) / divisor)
                        peaks_neg, props_neg = scipy.signal.find_peaks(data_ch * -1, distance=dist, height=np.max(data_ch * -1) / divisor)

                        # Assign number of peaks and troughs to the dataframe
                        # Only write if there was a legitimate component
                        if flag is False:
                            df_pt.at[subject, f'{data_type}_peaks_{cond_name}'] = len(peaks_pos)
                            df_pt.at[subject, f'{data_type}_troughs_{cond_name}'] = len(peaks_neg)

                        # Plot and save image with peaks and troughs marked
                        fig, ax = plt.subplots(1, 1)
                        ax.plot(evoked_ch.times, evoked_ch.get_data().reshape(-1), label='HF_CNAP')
                        ax.set_title(f'{subject_id}, High Freq, 400-800Hz')
                        ax.plot(evoked_ch.times[peaks_pos], data_ch[peaks_pos], '.', color='red', label='peaks')
                        ax.plot(evoked_ch.times[peaks_neg], data_ch[peaks_neg], '.', color='blue', label='troughs')
                        ax.axvline(ref_lat, color='green', label='low_freq_latency')
                        if data_type in ['bipolar', 'spinal']:
                            ax.set_xlim([0 / 1000, 30 / 1000])
                        else:
                            ax.set_xlim([0 / 1000, 50 / 1000])
                        plt.legend()
                        if data_type == 'bipolar':
                            plt.savefig(figure_path + f'{subject_id}_{channel[0]}_{excel_sheetname}')
                        else:
                            plt.savefig(figure_path + f'{subject_id}_{cond_name}_{channel}_{excel_sheetname}')
                        plt.close()

            with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
                df_pt.to_excel(writer, sheet_name=excel_sheetname)