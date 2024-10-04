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

    save_folder = f"/data/pt_02718/{folder}/WaveletCount_EqualWindow_JASPFormat/"
    os.makedirs(save_folder, exist_ok=True)

    for threshold in ['10%', '20%', '25%', '33%', '50%']:
        df_og = pd.read_excel(io=f"/data/pt_02718/{folder}/Peaks_Troughs_EqualWindow.xlsx", sheet_name=threshold)
        df_new = pd.DataFrame()
        df_new['Subject'] = df_og['Subject']

        for data_type in ['spinal', 'subcortical', 'cortical']:
            for cond_name in cond_names:
                df_new[f'{data_type}_{cond_name}_wavelets'] = df_og[[f'{data_type}_peaks_{cond_name}', f'{data_type}_troughs_{cond_name}']].mean(axis=1)

        df_new.set_index('Subject', inplace=True)
        with pd.ExcelWriter(save_folder + f"{threshold}.ods", engine="odf") as writer:
            df_new.to_excel(writer)