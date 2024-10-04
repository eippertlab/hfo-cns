###########################################################################################################
# Formatting previously configured exel files so I can use them with JASP
###########################################################################################################
import os

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

if __name__ == '__main__':
    srmr_nr = 2

    if srmr_nr == 1:
        folder = 'tmp_data'
        cond_names = ['median', 'tibial']
    elif srmr_nr == 2:
        folder = 'tmp_data_2'
        cond_names = ['med_mixed', 'tib_mixed']

    save_folder = f"/data/pt_02718/{folder}/PeakFrequency_JASPFormat/"
    os.makedirs(save_folder, exist_ok=True)


    df_new = pd.DataFrame()
    for data_type in ['Spinal', 'Thalamic', 'Cortical']:
        df_og = pd.read_excel(io=f"/data/pt_02718/{folder}/Peak_Frequency_400_800_ccabroadband_filter.xlsx",
                              sheet_name=data_type)
        df_new['Subject'] = df_og['Subject']

        for cond_name in cond_names:
                df_new[f'{data_type}_{cond_name}_peakfrequency'] = df_og[f'Peak_Frequency_{cond_name}']

    df_new.set_index('Subject', inplace=True)
    with pd.ExcelWriter(save_folder + f"burst_frequency.ods", engine="odf") as writer:
        df_new.to_excel(writer)