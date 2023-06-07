# Get the timing of the evoked potentials for our dataset

import pandas as pd
from scipy.io import loadmat
import numpy as np


def get_time_to_align(data_type, srmr_nr, cond_names, subjects):
    median = []
    tibial = []

    for cond_name in cond_names:
        if data_type == 'eeg':
            if srmr_nr == 1:
                xls = pd.ExcelFile('/data/pt_02718/tmp_data/Cortical_Timing.xlsx')
                df_timing = pd.read_excel(xls, 'Timing')
                df_timing.set_index('Subject', inplace=True)
            elif srmr_nr == 2:
                xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Cortical_Timing.xlsx')
                df_timing = pd.read_excel(xls, 'Timing')
                df_timing.set_index('Subject', inplace=True)

        elif data_type == 'esg':
            if srmr_nr == 1:
                xls = pd.ExcelFile('/data/pt_02718/tmp_data/Spinal_Timing.xlsx')
                df_timing = pd.read_excel(xls, 'Timing')
                df_timing.set_index('Subject', inplace=True)
            elif srmr_nr == 2:
                xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Spinal_Timing.xlsx')
                df_timing = pd.read_excel(xls, 'Timing')
                df_timing.set_index('Subject', inplace=True)

        for subject in subjects:
            if data_type == 'esg':
                if cond_name in ['median', 'med_mixed']:
                    sep_latency = df_timing.loc[subject, f"N13"]
                    median.append(sep_latency)
                elif cond_name in ['tibial', 'tib_mixed']:
                    sep_latency = df_timing.loc[subject, f"N22"]
                    tibial.append(sep_latency)
            elif data_type == 'eeg':
                if cond_name in ['median', 'med_mixed']:
                    sep_latency = df_timing.loc[subject, f"N20"]
                    median.append(sep_latency)
                elif cond_name in ['tibial', 'tib_mixed']:
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
    median_lat, tibial_lat = get_time_to_align('eeg', 1, ['median', 'tibial'], np.arange(1, 37))
    # median_lat, tibial_lat = get_time_to_align('eeg', 2, ['med_mixed', 'tib_mixed'], np.arange(1, 25))
    print(median_lat)
    print(tibial_lat)
    print('ESG')
    median_lat, tibial_lat = get_time_to_align('esg', 1, ['median', 'tibial'], np.arange(1, 37))
    # median_lat, tibial_lat = get_time_to_align('esg', 2, ['med_mixed', 'tib_mixed'], np.arange(1, 25))
    print(median_lat)
    print(tibial_lat)
