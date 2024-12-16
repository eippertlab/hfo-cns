import h5py
import numpy as np
import os
import mne
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.evoked_from_raw import evoked_from_raw

if __name__ == '__main__':
    # Read in the data before filtering and check the amplitude of the evoked low-freq response

    data_types = ['Spinal']  # Only for spinal now
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

    for condition in conditions:
        amplitudes = []
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        if cond_name in ['tibial', 'tib_mixed']:
            channel = ['L1']
            start = 19 / 1000
            end = 25 / 1000

        elif cond_name in ['median', 'med_mixed']:
            channel = ['SC6']
            start = 10 / 1000
            end = 16 / 1000

        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            file_path = f"/data/pt_02718/{folder}/ssp_cleaned/{subject_id}/ssp6_cleaned_{cond_name}.fif"

            raw = mne.io.read_raw_fif(file_path, preload=True)
            # n_channels, n_times
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked_ch = evoked.pick(channel)

            ch_name, latency, amplitude = evoked_ch.get_peak(tmin=start, tmax=end, mode='neg', return_amplitude=True,
                                                             strict=False)
            amplitudes.append(amplitude)

        print(f"{cond_name}: {np.mean(amplitudes)}")