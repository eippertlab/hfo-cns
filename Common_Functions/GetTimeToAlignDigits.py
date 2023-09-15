# Get the timing of the evoked potentials for our dataset

import pandas as pd
from scipy.io import loadmat
import numpy as np


def get_time_to_align_digits(data_type, srmr_nr, cond_names, subjects):
    median = []
    tibial = []

    for cond_name in cond_names:
        if data_type == 'eeg':
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Cortical_Timing_Digits.xlsx')
            df_timing = pd.read_excel(xls, 'Timing')
            df_timing.set_index('Subject', inplace=True)

        elif data_type == 'esg':
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Spinal_Timing_Digits.xlsx')
            df_timing = pd.read_excel(xls, 'Timing')
            df_timing.set_index('Subject', inplace=True)

        for subject in subjects:
            if data_type == 'esg':
                if cond_name == 'med_digits':
                    sep_latency = df_timing.loc[subject, f"N13"]
                    median.append(sep_latency)
                elif cond_name == 'tib_digits':
                    sep_latency = df_timing.loc[subject, f"N22"]
                    tibial.append(sep_latency)
            elif data_type == 'eeg':
                if cond_name == 'med_digits':
                    sep_latency = df_timing.loc[subject, f"N20"]
                    median.append(sep_latency)
                elif cond_name == 'tib_digits':
                    sep_latency = df_timing.loc[subject, f"P39"]
                    tibial.append(sep_latency)

    # print(f"Median: {np.mean(median)}")
    # print(f"Tibial: {np.mean(tibial)}")
    # print(median)
    # print(tibial)
    # Want it rounded to the nearest ms
    return round(np.mean(median), 3), round(np.mean(tibial), 3)


if __name__ == '__main__':
    print('EEG')
    median_lat, tibial_lat = get_time_to_align_digits('eeg', 1, ['med_digits', 'tib_digits'], np.arange(1, 25))
    print(median_lat)
    print(tibial_lat)
    print('ESG')
    median_lat, tibial_lat = get_time_to_align_digits('esg', 1, ['med_digits', 'tib_digits'], np.arange(1, 25))
    print(median_lat)
    print(tibial_lat)
