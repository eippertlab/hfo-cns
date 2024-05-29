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
    srmr_nr = 1
    high_freq = 800
    ttest_type = 'med_v_tib' # 'across_cns' 'med_v_tib'

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

    # Compute difference between CNS levels for median and tibial nerve stimulation
    if ttest_type == 'across_cns':
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

            # If rm_anova was significant, perform post-hoc t-tests
            if aov['p-unc'].loc[aov.index[0]] < 0.05:
                # Two-sided
                ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='two-sided')
                print(ptest_mat)

                # Less than
                ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='less')
                print(ptest_mat)

    elif ttest_type == 'med_v_tib':
        for data_type in data_types:
            df_combination = pd.DataFrame()
            # df_combination['Subject'] = subjects
            sheetname = data_type
            df_freq = pd.read_excel(excel_fname, sheetname)
            df_combination[f"{data_type}_{cond_names[0]}"] = df_freq[f'Peak_Frequency_{cond_names[0]}']
            df_combination[f"{data_type}_{cond_names[1]}"] = df_freq[f'Peak_Frequency_{cond_names[1]}']

            df_combination.dropna(inplace=True)
            print(df_combination.describe())
            print(df_combination.sem())

            # Two-sided
            ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='two-sided')
            print('two-sided')
            print(ptest_mat)

            # Less than
            ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='less')
            print('less')
            print(ptest_mat)

            # More than
            ptest_mat = df_combination.ptests(paired=True, padjust='bonf', stars=False, alternative='greater')
            print('greater')
            print(ptest_mat)

