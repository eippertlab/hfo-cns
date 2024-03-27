# Look at the ratio of HFO peak closest to low freq peak in relation to closest two positive and negative peaks

import mne
from Common_Functions.get_conditioninfo import get_conditioninfo
import numpy as np
import pandas as pd
from Common_Functions.check_excel_exist_general import check_excel_exist_general
import matplotlib.pyplot as plt
import os
import scipy
from Common_Functions.invert import invert
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

    sampling_rate = 5000
    sampling_rate_bipolar = 10000

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    notch_freq = df.loc[df['var_name'] == 'notch_freq', 'var_value'].iloc[0]

    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    # Set up dataframe columns
    if srmr_nr == 1:
        add = ''
        col_names = ['Subject',
                     'central_lat_median', 'central_amp_median', 'ratio_same_median', 'ratio_diff_median',
                     'central_lat_tibial', 'central_amp_tibial', 'ratio_same_tibial', 'ratio_diff_tibial']
    elif srmr_nr == 2:
        add = '_2'
        col_names = ['Subject',
                     'central_lat_med_mixed', 'central_amp_med_mixed', 'ratio_same_med_mixed', 'ratio_diff_med_mixed',
                     'central_lat_tib_mixed', 'central_amp_tib_mixed', 'ratio_same_tib_mixed', 'ratio_diff_tib_mixed']

    save_path = f"/data/pt_02718/tmp_data{add}/"
    figure_path = f'/data/p_02718/Images{add}/HFO_TimeCourse_Peaks/'
    os.makedirs(figure_path, exist_ok=True)

    time_edge_med = 3/1000
    time_edge_tib = 6/1000

    for data_type in ['spinal', 'cortical']:
        # Excel for saving result
        excel_fname = f'{save_path}HF_CortSpin_Ratio.xlsx'
        excel_sheetname = f'{data_type}'
        check_excel_exist_general(subjects, fname=excel_fname, sheetname=excel_sheetname, col_names=col_names)
        df_latency = pd.read_excel(excel_fname, excel_sheetname)
        df_latency.set_index('Subject', inplace=True)

        # Read in the low frequency latency so I can get those to use as reference to find the ratios of the HFOs
        excel_fname_load = f'/data/pt_02718/tmp_data{add}/LowFreq_HighFreq_Relation.xlsx'
        excel_sheetname_load = data_type.capitalize()
        df_lat_low = pd.read_excel(excel_fname_load, excel_sheetname_load)

        if data_type == 'spinal':
            low_latency_med = df_lat_low[f'N13'].to_list()
            low_latency_tib = df_lat_low[f'N22'].to_list()
            # For checking the subject has a cca component
            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{add}/Components_Updated.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{add}/Visibility_Updated.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Spinal')
            df_vis.set_index('Subject', inplace=True)

        elif data_type == 'cortical':
            low_latency_med = df_lat_low[f'N20'].to_list()
            low_latency_tib = df_lat_low[f'P39'].to_list()
            # For checking the subject has a cca component
            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{add}/Components_EEG_Updated.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            xls = pd.ExcelFile(f'/data/pt_02718/tmp_data{add}/Visibility_Updated.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

        for condition in conditions:
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            stimulation = condition - 1

            for subject in subjects:
                visible = df_vis.loc[subject, f"Sigma_{cond_name.capitalize()}_Visible"]
                # Set paths
                subject_id = f'sub-{str(subject).zfill(3)}'

                if visible == 'T':
                    fname = f"sigma_{cond_name}.fif"
                    if data_type == 'spinal':
                        input_path = f"/data/pt_02718/tmp_data{add}/cca/{subject_id}/"
                    elif data_type == 'cortical':
                        input_path = f"/data/pt_02718/tmp_data{add}/cca_eeg/{subject_id}/"

                    if cond_name in ['median', 'med_mixed']:
                        ref_lat = low_latency_med[subject-1]
                        time_edge = time_edge_med
                    else:
                        ref_lat = low_latency_tib[subject-1]
                        time_edge = time_edge_tib

                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    channel_no = df.loc[subject, f"sigma_{cond_name}_comp"]
                    channels = f'Cor{channel_no}'
                    inv = df.loc[subject, f"sigma_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channels])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channels)
                    evoked_ch = epochs.copy().average()
                    data_ch = evoked_ch.get_data().reshape(-1)

                    # Get pos and neg peaks
                    peaks_pos, props_pos = scipy.signal.find_peaks(data_ch, distance=1, height=np.max(data_ch)/7)
                    peaks_neg, props_neg = scipy.signal.find_peaks(data_ch*-1, distance=1, height=np.max(data_ch*-1)/7)

                    # Find pos and neg peaks that are within time window of interest
                    # Convert time window to sample numbers
                    edge_before = find_nearest(evoked_ch.times, ref_lat-time_edge)
                    index_edge_before = list(evoked_ch.times).index(edge_before)
                    edge_after = find_nearest(evoked_ch.times, ref_lat+time_edge)
                    index_edge_after = list(evoked_ch.times).index(edge_after)

                    # Convert allowed peak latencies to their indices
                    index_central_pos = [index for index in peaks_pos if index_edge_before < index < index_edge_after]
                    index_central_neg = [index for index in peaks_neg if index_edge_before < index < index_edge_after]

                    # Get index of pos and neg peak with maximum amplitude
                    pos_max_peak = index_central_pos[np.argmax(data_ch[index_central_pos])]
                    neg_max_peak = index_central_neg[np.argmin(data_ch[index_central_neg])]

                    # Find timing and amplitude from this peak after determining whether peak/trough is absolute min/max
                    # in time zone of interest
                    if abs(data_ch[pos_max_peak]) > abs(data_ch[neg_max_peak]):
                        lat_centre = evoked_ch.times[pos_max_peak]
                        index_centre = list(evoked_ch.times).index(lat_centre)
                        amp_centre = data_ch[index_centre]
                        peak = 'pos'
                    else:
                        lat_centre = evoked_ch.times[neg_max_peak]
                        index_centre = list(evoked_ch.times).index(lat_centre)
                        amp_centre = data_ch[index_centre]
                        peak = 'neg'

                    # Calculate ratio to the two closest peaks of the same polarity and the two closest peaks of the
                    # opposite polarity
                    if peak == 'neg':
                        index_left_same = [x for x in peaks_neg if x < index_centre][-1]
                        lat_left_same = evoked_ch.times[index_left_same]
                        amp_left_same = data_ch[index_left_same]

                        index_right_same = [x for x in peaks_neg if x > index_centre][0]
                        lat_right_same = evoked_ch.times[index_right_same]
                        amp_right_same = data_ch[index_right_same]

                        index_left_diff = [x for x in peaks_pos if x < index_centre][-1]
                        lat_left_diff = evoked_ch.times[index_left_diff]
                        amp_left_diff = data_ch[index_left_diff]

                        index_right_diff = [x for x in peaks_pos if x > index_centre][0]
                        lat_right_diff = evoked_ch.times[index_right_diff]
                        amp_right_diff = data_ch[index_right_diff]

                    elif peak == 'pos':
                        index_left_same = [x for x in peaks_pos if x < index_centre][-1]
                        lat_left_same = evoked_ch.times[index_left_same]
                        amp_left_same = data_ch[index_left_same]

                        index_right_same = [x for x in peaks_pos if x > index_centre][0]
                        lat_right_same = evoked_ch.times[index_right_same]
                        amp_right_same = data_ch[index_right_same]

                        index_left_diff = [x for x in peaks_neg if x < index_centre][-1]
                        lat_left_diff = evoked_ch.times[index_left_diff]
                        amp_left_diff = data_ch[index_left_diff]

                        index_right_diff = [x for x in peaks_neg if x > index_centre][0]
                        lat_right_diff = evoked_ch.times[index_right_diff]
                        amp_right_diff = data_ch[index_right_diff]

                    # Assign to the dataframe
                    df_latency.at[subject, f'central_lat_{cond_name}'] = lat_centre*1000
                    df_latency.at[subject, f'central_amp_{cond_name}'] = amp_centre
                    df_latency.at[subject, f'ratio_same_{cond_name}'] = abs(amp_centre)/abs(np.mean([amp_right_same, amp_left_same]))
                    df_latency.at[subject, f'ratio_diff_{cond_name}'] = abs(amp_centre)/abs(np.mean([amp_right_diff, amp_left_diff]))

                    # Plot and save image with peaks and troughs marked
                    fig, ax = plt.subplots(1, 1)
                    ax.plot(evoked_ch.times, evoked_ch.get_data().reshape(-1), label='HF')
                    ax.set_title(f'{subject_id}, High Freq, 400-800Hz')
                    ax.plot(evoked_ch.times[peaks_pos], data_ch[peaks_pos], '.', color='red', label='peaks')
                    ax.plot(evoked_ch.times[peaks_neg], data_ch[peaks_neg], '.', color='blue', label='troughs')
                    ax.axvline(ref_lat, color='green', label='low_freq_latency')
                    ax.axvline(lat_centre, color='black', label='peak_used')
                    ax.set_xlim([0 / 1000, 70 / 1000])
                    plt.legend()
                    plt.savefig(figure_path + f'{data_type}_{subject_id}_{channels}_{cond_name}')
                    plt.close()

        with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
            df_latency.to_excel(writer, sheet_name=excel_sheetname)