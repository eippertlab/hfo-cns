# Get the timing of the evoked potentials for our dataset

import pandas as pd
from scipy.io import loadmat
import numpy as np


def get_time_to_align(data_type, cond_names, subjects):
    median = []
    tibial = []

    for cond_name in cond_names:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'

            if data_type == 'eeg':
                xls = pd.ExcelFile('/data/pt_02718/tmp_data/Cortical_Timing.xlsx')
                df_timing = pd.read_excel(xls, 'Timing')
                df_timing.set_index('Subject', inplace=True)
                if cond_name == 'median':
                    sep_latency = df_timing.loc[subject, f"N20"]
                    median.append(sep_latency)
                elif cond_name == 'tibial':
                    sep_latency = df_timing.loc[subject, f"P39"]
                    tibial.append(sep_latency)

            elif data_type == 'esg':
                xls = pd.ExcelFile('/data/pt_02718/tmp_data/Spinal_Timing.xlsx')
                df_timing = pd.read_excel(xls, 'Timing')
                df_timing.set_index('Subject', inplace=True)
                if cond_name == 'median':
                    sep_latency = df_timing.loc[subject, f"N13"]
                    median.append(sep_latency)
                elif cond_name == 'tibial':
                    sep_latency = df_timing.loc[subject, f"N22"]
                    tibial.append(sep_latency)

    # print(f"Median: {np.mean(median)}")
    # print(f"Tibial: {np.mean(tibial)}")
    # Want it rounded to the nearest ms
    return round(np.mean(median), 3), round(np.mean(tibial), 3)


if __name__ == '__main__':
    print('EEG')
    median_lat, tibial_lat = get_time_to_align('eeg', ['median', 'tibial'], np.arange(1, 37))
    print(median_lat)
    print(tibial_lat)
    print('ESG')
    median_lat, tibial_lat = get_time_to_align('esg', ['median', 'tibial'], np.arange(1, 37))
    print(median_lat)
    print(tibial_lat)
