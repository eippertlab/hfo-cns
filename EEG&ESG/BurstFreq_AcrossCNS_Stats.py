# Plotting some initial images about the relation between frequency in different subjects


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import pingouin as pg
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 2
    high_freq = 800

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        data_types = ['Spinal', 'Thalamic', 'Cortical']
        excel_fname = f'/data/pt_02718/tmp_data/Peak_Frequency_400_{high_freq}_ccabroadband_filter.xlsx'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        data_types = ['Spinal', 'Thalamic', 'Cortical']
        excel_fname = f'/data/pt_02718/tmp_data_2/Peak_Frequency_400_{high_freq}_ccabroadband_filter.xlsx'

    for cond_name in cond_names:
        df_combination = pd.DataFrame()
        # df_combination['Subject'] = subjects
        col_name = f'Peak_Frequency_{cond_name}'
        for data_type in data_types:
            sheetname = data_type
            df_freq = pd.read_excel(excel_fname, sheetname)
            df_combination[f"{data_type}"] = df_freq[col_name]

        df_combination.dropna(inplace=True)
        print(cond_name)
        print(df_combination.describe())
        print(df_combination.sem())

        aov = df_combination.rm_anova()
        print(aov)

        # df_combination = pd.melt(df_combination, id_vars=['Subject'],
        #                          value_vars=[f'Spinal', f'Thalamic', f'Cortical'],
        #                          var_name='CNS Level', value_name='Frequency')  # Change to long format

        # If rm_anova was significant, perform post-hoc t-tests
        if aov['p-unc'].loc[aov.index[0]] < 0.05:
            # Two-sided
            ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='two-sided')
            print(ptest_mat)

            # Less than
            ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='less')
            print(ptest_mat)

