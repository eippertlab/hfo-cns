# Get the latency of the low frequency CNAP at the biceps and kneem electrode
# N6 in biceps (6.22ms latency for Birgit), N8 in knee (9.28ms latency for birgit)

import mne
from Common_Functions.get_conditioninfo import get_conditioninfo
import numpy as np
import pandas as pd
from Common_Functions.check_excel_exist_general import check_excel_exist_general
import matplotlib.pyplot as plt
import os


if __name__ == '__main__':
    srmr_nr = 2
    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [3, 2]
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]

    sampling_rate_og = 10000

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
        figure_path = '/data/p_02718/Images/Bipolar_Images/WidebandTimeCourse_WithLatency/'
        os.makedirs(figure_path, exist_ok=True)
        col_names = ['Subject', 'lat_median', 'amp_median', 'lat_tibial', 'amp_tibial']
    elif srmr_nr == 2:
        save_path = f"/data/pt_02718/tmp_data_2/"
        figure_path = '/data/p_02718/Images_2/Bipolar_Images/WidebandTimeCourse_WithLatency/'
        os.makedirs(figure_path, exist_ok=True)
        col_names = ['Subject', 'lat_med_mixed', 'amp_med_mixed', 'lat_tib_mixed', 'amp_tib_mixed']
    excel_fname = f'{save_path}Bipolar_Latency.xlsx'
    excel_sheetname = 'LowFrequency'
    check_excel_exist_general(subjects, fname=excel_fname, sheetname=excel_sheetname, col_names=col_names)
    df_latency = pd.read_excel(excel_fname, excel_sheetname)
    df_latency.set_index('Subject', inplace=True)

    for condition in conditions:
        timing = []
        amp = []

        for subject in subjects:

            # Set paths
            subject_id = f'sub-{str(subject).zfill(3)}'
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            stimulation = condition - 1
            if srmr_nr == 1:
                input_path = f"/data/pt_02718/tmp_data/imported/{subject_id}/"
            elif srmr_nr == 2:
                input_path = f"/data/pt_02718/tmp_data_2/imported/{subject_id}/"
            fname = f'bipolar_repaired_{cond_name}.fif'

            if cond_name in ['median', 'med_mixed']:
                channel = ['Biceps']
                expected = 6/1000
                time_edge_neg = 1.5/1000
                time_edge_pos = 2/1000

            elif cond_name in ['tibial', 'tib_mixed']:
                channel = ['KneeM']
                expected = 9/1000
                time_edge_neg = 2/1000
                time_edge_pos = 3/1000

            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            raw.filter(30, 400, picks=channel)

            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline))
            evoked = epochs.average(picks='all')

            evoked_ch = evoked.pick(channel)
            data_low = evoked_ch.copy().crop(tmin=expected - time_edge_neg, tmax=expected + time_edge_pos).get_data().reshape(
                -1)
            if min(data_low) < 0:
                _, latency, amplitude = evoked_ch.get_peak(tmin=expected-time_edge_neg, tmax=expected+time_edge_pos,
                                                           mode='neg', return_amplitude=True)
            else:
                latency = np.nan
                amplitude = np.nan
            timing.append(latency*1000)
            amp.append(amplitude*10**6)

            fig, ax = plt.subplots(1, 1)
            ax.plot(evoked.times, evoked.get_data(picks=channel).reshape(-1))
            ax.set_title('Wideband')
            if latency is not np.nan:
                ax.axvline(latency, color='red')
            ax.set_xlim([-20 / 1000, 30 / 1000])
            plt.savefig(figure_path+f'{subject_id}_{channel[0]}')
            plt.close()

        df_latency.loc[:, f'lat_{cond_name}'] = timing
        df_latency.loc[:, f'amp_{cond_name}'] = amp

    with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
        df_latency.to_excel(writer, sheet_name=excel_sheetname)
