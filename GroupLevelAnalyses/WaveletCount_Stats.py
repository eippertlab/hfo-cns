# Test the difference in wavelet count across the CNS at the different threshold levels

import numpy as np
import pingouin as pg
import pandas as pd
import matplotlib as mpl
import os
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        excel_fname = '/data/pt_02718/tmp_data/Peaks_Troughs_EqualWindow.xlsx'
        figure_path = f'/data/p_02718/Images/Peak_Trough_Images_EqualWindow/CrossCNS_RankCorrelation/'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        excel_fname = '/data/pt_02718/tmp_data_2/Peaks_Troughs_EqualWindow.xlsx'
        figure_path = '/data/p_02718/Images_2/Peak_Trough_Images_EqualWindow/CrossCNS_RankCorrelation/'
    os.makedirs(figure_path, exist_ok=True)
    data_types = ['spinal', 'subcortical', 'cortical']

    for cond_name in cond_names:
        for sheetname in ['10%', '20%', '25%', '33%', '50%']:
            df_pt = pd.read_excel(excel_fname, sheetname)
            col_names_troughs = [f'spinal_troughs_{cond_name}', f'subcortical_troughs_{cond_name}', f'cortical_troughs_{cond_name}']
            col_names_peaks = [f'spinal_peaks_{cond_name}', f'subcortical_peaks_{cond_name}', f'cortical_peaks_{cond_name}']

            # Need to get average of peaks and troughs for each subject and then make new df with these values
            df_wavelet = pd.DataFrame()
            df_wavelet['Subject'] = subjects
            for data_type in data_types:
                df_wavelet[f'{data_type.capitalize()}'] = df_pt[[f'{data_type}_peaks_{cond_name}', f'{data_type}_troughs_{cond_name}']].mean(axis=1)
            print(f'{cond_name}, {sheetname}')
            df_wavelet.dropna(inplace=True)
            df_longform = df_wavelet.melt(id_vars=['Subject'], var_name='CNS_Level', value_name='no_wavelets')


            print('Friedman Test')
            print(pg.friedman(df_wavelet, method='f'))
            # Perform the Friedman test
            # stat, p = friedmanchisquare(df_wavelet['Spinal'], df_wavelet['Subcortical'], df_wavelet['Cortical'])
            # print(f'Friedman test statistic={stat:.3f}, p-value={p:.3f}')

            if pg.friedman(df_wavelet)['p-unc'][0] < 0.05:
                print('Post-hoc wilcoxon')
                result = pg.pairwise_tests(data=df_longform, dv='no_wavelets', subject='Subject', within='CNS_Level',
                                           parametric=False, alternative='two-sided', padjust='bonf')
                print(result)
                # Nemenyi test nt specific to repeated measures design - used Wilcoxon
                # nemenyi_results = sp.posthoc_nemenyi_friedman(df_wavelet)
                # print(nemenyi_results)
                # exit()
            print('\n\n')
