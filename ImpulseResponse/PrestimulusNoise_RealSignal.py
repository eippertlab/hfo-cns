# Want to look at the standard deviation in the pre-stimulus period of our signal
# So that we know what std of noise to add to the signal for the impulse response

import os
import mne
import numpy as np
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels

if __name__ == '__main__':
    data_types = ['Cortical', 'Spinal']
    srmr_nr = 2

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    for data_type in data_types:
        for condition in conditions:
            std_list = []
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            for subject in subjects:
                eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

                noise_window = [-100 / 1000, -10 / 1000]
                subject_id = f'sub-{str(subject).zfill(3)}'

                if data_type == 'Cortical':
                    # Wideband SEP
                    input_path = f"/data/pt_02718/{folder}/imported/{subject_id}/"
                    fname = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"

                elif data_type == 'Spinal':
                    # Wideband SEP
                    input_path = f"/data/pt_02718/{folder}/ssp_cleaned/{subject_id}/"
                    fname = f"ssp6_cleaned_{cond_name}.fif"

                raw = mne.io.read_raw_fif(input_path+fname, preload=True)
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                    baseline=tuple(iv_baseline), preload=True)

                # Want std of each channel across times, then average of this across all channels and trials
                epochs.crop(tmin=noise_window[0], tmax=noise_window[1])
                data = epochs.get_data()  # n_trials, n_channels, n_times
                sd = data.std(axis=2)  # std across times
                sd_avg = np.mean(sd, axis=tuple([0, 1]))  # Average across trials and channels
                std_list.append(sd_avg*10**6)

            print(f"{data_type}, {condition}, std: {np.mean(std_list)}")

