# Get the latency of the low frequency CNAP at the biceps and kneem electrode
# N6 in biceps (6.22ms latency for Birgit), N8 in knee (9.28ms latency for birgit)
# Must run low freq timing before this one so we have the reference latency **

import mne
from Common_Functions.get_conditioninfo import get_conditioninfo
import numpy as np
import pandas as pd
from Common_Functions.check_excel_exist_general import check_excel_exist_general
import matplotlib.pyplot as plt
import os
import scipy
import math
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


if __name__ == '__main__':
    srmr_nr = 2

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]

    sampling_rate = 10000

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    notch_freq = df.loc[df['var_name'] == 'notch_freq', 'var_value'].iloc[0]

    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    # Set up the dataframe to store results
    if srmr_nr == 1:
        save_path = f"/data/pt_02718/tmp_data/"
        figure_path = '/data/p_02718/Images/Bipolar_Images/HFO_TimeCourse_WithLatency/'
        os.makedirs(figure_path, exist_ok=True)
        col_names = ['Subject',
                     'central_lat_median', 'central_amp_median', 'ratio_neg_median', 'ratio_pos_median',
                     'central_lat_tibial', 'central_amp_tibial', 'ratio_neg_tibial', 'ratio_pos_tibial']
    elif srmr_nr == 2:
        save_path = f"/data/pt_02718/tmp_data_2/"
        figure_path = '/data/p_02718/Images_2/Bipolar_Images/HFO_TimeCourse_WithLatency/'
        os.makedirs(figure_path, exist_ok=True)
        col_names = ['Subject',
                     'central_lat_med_mixed', 'central_amp_med_mixed', 'ratio_neg_med_mixed', 'ratio_pos_med_mixed',
                     'central_lat_tib_mixed', 'central_amp_tib_mixed', 'ratio_neg_tib_mixed', 'ratio_pos_tib_mixed']

    excel_fname = f'{save_path}Bipolar_Latency.xlsx'
    excel_sheetname = 'HighFrequency'
    check_excel_exist_general(subjects, fname=excel_fname, sheetname=excel_sheetname, col_names=col_names)
    df_latency = pd.read_excel(excel_fname, excel_sheetname)
    df_latency.set_index('Subject', inplace=True)

    # Read in the low frequency latency so I can get those to use as reference to find the ratios of the HFOs
    df_lat_low = pd.read_excel(excel_fname, 'LowFrequency')
    if srmr_nr == 1:
        low_latency_med = df_lat_low[f'lat_median'].to_list()
        low_latency_tib = df_lat_low[f'lat_tibial'].to_list()
    elif srmr_nr == 2:
        low_latency_med = df_lat_low[f'lat_med_mixed'].to_list()
        low_latency_tib = df_lat_low[f'lat_tib_mixed'].to_list()

    for condition in conditions:
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name
        stimulation = condition - 1

        for subject in subjects:
            flag = False
            # Set paths
            subject_id = f'sub-{str(subject).zfill(3)}'

            if cond_name in ['median', 'med_mixed']:
                ref_lat = low_latency_med[subject-1]/1000
                if math.isnan(ref_lat):
                    ref_lat = 6/1000
                    flag=True
            else:
                ref_lat = low_latency_tib[subject-1]/1000
                if math.isnan(ref_lat):
                    ref_lat = 9/1000
                    flag=True

            if srmr_nr == 1:
                input_path = f"/data/pt_02718/tmp_data/freq_banded_bipolar/{subject_id}/"
            elif srmr_nr == 2:
                input_path = f"/data/pt_02718/tmp_data_2/freq_banded_bipolar/{subject_id}/"
            fname = f'sigma_{cond_name}.fif'

            if cond_name in ['median', 'med_mixed']:
                channel = ['Biceps']
            elif cond_name in ['tibial', 'tib_mixed']:
                channel = ['KneeM']

            raw = mne.io.read_raw_fif(input_path + fname, preload=True)

            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline))
            evoked = epochs.average(picks='all')
            evoked_ch = evoked.pick(channel).crop(0/1000, 30/1000)

            data_ch = evoked_ch.get_data().reshape(-1)

            # Get pos and neg peaks
            peaks_pos, props_pos = scipy.signal.find_peaks(data_ch, distance=1, height=np.max(data_ch)/7)
            peaks_neg, props_neg = scipy.signal.find_peaks(data_ch*-1, distance=1, height=np.max(data_ch*-1)/7)

            # Use the latency of the low frequency NAP to get the latency and amplitude of the negative peak and the
            # closest two neg and pos peaks on either side
            # Only perform this if we have a reference latency from the low freq data
            # Central Negativity
            try:
                lat_centre_neg = find_nearest(evoked_ch.times[peaks_neg], ref_lat)
                index_centre_neg = list(evoked_ch.times).index(lat_centre_neg)
                amp_centre_neg = data_ch[index_centre_neg]

                index_left_neg = [x for x in peaks_neg if x < index_centre_neg][-1]
                lat_left_neg = evoked_ch.times[index_left_neg]
                amp_left_neg = data_ch[index_left_neg]

                index_right_neg = [x for x in peaks_neg if x > index_centre_neg][0]
                lat_right_neg = evoked_ch.times[index_right_neg]
                amp_right_neg = data_ch[index_right_neg]

                index_left_pos = [x for x in peaks_pos if x < index_centre_neg][-1]
                lat_left_pos = evoked_ch.times[index_left_pos]
                amp_left_pos = data_ch[index_left_pos]

                index_right_pos = [x for x in peaks_pos if x > index_centre_neg][0]
                lat_right_pos = evoked_ch.times[index_right_pos]
                amp_right_pos = data_ch[index_right_pos]

                # Assign to the dataframe
                df_latency.at[subject, f'central_lat_{cond_name}'] = lat_centre_neg*1000
                df_latency.at[subject, f'central_amp_{cond_name}'] = amp_centre_neg*10**6
                df_latency.at[subject, f'ratio_neg_{cond_name}'] = abs(amp_centre_neg)/abs(np.mean([amp_right_neg, amp_left_neg]))
                df_latency.at[subject, f'ratio_pos_{cond_name}'] = abs(amp_centre_neg)/abs(np.mean([amp_right_pos, amp_left_pos]))

            except:
                # Assign to the dataframe nan if something goes wrong
                df_latency.at[subject, f'central_lat_{cond_name}'] = np.nan
                df_latency.at[subject, f'central_amp_{cond_name}'] = np.nan
                df_latency.at[subject, f'ratio_neg_{cond_name}'] = np.nan
                df_latency.at[subject, f'ratio_pos_{cond_name}'] = np.nan

            # Plot and save image with peaks and troughs marked
            fig, ax = plt.subplots(1, 1)
            ax.plot(evoked_ch.times, evoked_ch.get_data().reshape(-1), label='HF_CNAP')
            if flag is True:
                ax.set_title(f'{subject_id}, High Freq, 400-800Hz, False Ref Lat')
            else:
                ax.set_title(f'{subject_id}, High Freq, 400-800Hz, True Ref Lat')
            ax.plot(evoked_ch.times[peaks_pos], data_ch[peaks_pos], '.', color='red', label='peaks')
            ax.plot(evoked_ch.times[peaks_neg], data_ch[peaks_neg], '.', color='blue', label='troughs')
            ax.axvline(ref_lat, color='green', label='low_freq_latency')
            ax.set_xlim([0 / 1000, 30 / 1000])
            plt.legend()
            plt.savefig(figure_path + f'{subject_id}_{channel[0]}')
            plt.close()

    with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
        df_latency.to_excel(writer, sheet_name=excel_sheetname)