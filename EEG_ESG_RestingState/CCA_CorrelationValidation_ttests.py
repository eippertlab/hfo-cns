###################################################################################
# We have average correlation per participant, condition (median/tibial), and task (task/resting state), and CNS level
# (cortical, subcortical, spinal)
# We want to test the across subject, within condition, across task differences at each CNS level
# i.e. do median, task differ from median, rest in the spinal data


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import pingouin as pg

if __name__ == '__main__':
    srmr_nr = 2
    type = 'cca'  # Can be long, shorter or cca

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        if type == 'long':
            excel_fname = f'/data/pt_02718/tmp_data/Correlation_Long.xlsx'
        elif type == 'shorter':
            excel_fname = f'/data/pt_02718/tmp_data/Correlation_Shorter.xlsx'
        elif type == 'cca':
            excel_fname = f'/data/pt_02718/tmp_data/Correlation_CCAWin.xlsx'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        if type == 'long':
            excel_fname = f'/data/pt_02718/tmp_data_2/Correlation_Long.xlsx'
        elif type == 'shorter':
            excel_fname = f'/data/pt_02718/tmp_data_2/Correlation_Shorter.xlsx'
        elif type == 'cca':
            excel_fname = f'/data/pt_02718/tmp_data_2/Correlation_CCAWin.xlsx'

    data_types = ['spinal', 'subcortical', 'cortical']
    sheetname = 'Correlation'
    df = pd.read_excel(excel_fname, sheetname)
    df.drop('Subject', axis=1, inplace=True)
    print(df.mean())
    print(df.sem())

    # Test just the 12 tests we're interested in and correct them
    dict_pvals = {}
    for data_type in data_types:
        for cond_name in cond_names:
            df_totest = df[[f'{data_type}_{cond_name}_task', f'{data_type}_{cond_name}_rest']]
            # print(df_totest)
            stats = df_totest.ptests(paired=True, stars=False, decimals=30, alternative='greater')
            p_val = stats.loc[f'{data_type}_{cond_name}_task', f'{data_type}_{cond_name}_rest']
            dict_pvals[f'{data_type}_{cond_name}'] = float(p_val)

    for key in dict_pvals.keys():
        print(f'{key}: {dict_pvals[key]}')