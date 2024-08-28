# Plotting how the number of peaks and troughs varies across levels of the CNS for each subject


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        excel_fname = '/data/pt_02718/tmp_data/Peaks_Troughs_EqualWindow.xlsx'
        figure_path = f'/data/p_02718/Images/Peak_Trough_Images_EqualWindow/CrossCNS_Change_ByCutoff/'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        excel_fname = '/data/pt_02718/tmp_data_2/Peaks_Troughs_EqualWindow.xlsx'
        figure_path = '/data/p_02718/Images_2/Peak_Trough_Images_EqualWindow/CrossCNS_Change_ByCutoff/'
    os.makedirs(figure_path, exist_ok=True)
    data_types = ['spinal', 'subcortical', 'cortical']

    for sheetname in ['10%', '20%', '25%', '33%', '50%']:
    # for sheetname in ['33%']:
        df_pt = pd.read_excel(excel_fname, sheetname)
        for cond_name in cond_names:
            col_names_troughs = [f'spinal_troughs_{cond_name}', f'subcortical_troughs_{cond_name}', f'cortical_troughs_{cond_name}']
            col_names_peaks = [f'spinal_peaks_{cond_name}', f'subcortical_peaks_{cond_name}', f'cortical_peaks_{cond_name}']

            # Need to get average of peaks and troughs for each subject and then make new df with these values
            df_wavelet = pd.DataFrame()
            df_wavelet['Subject'] = subjects
            for data_type in data_types:
                df_wavelet[f'{data_type.capitalize()}'] = df_pt[[f'{data_type}_peaks_{cond_name}', f'{data_type}_troughs_{cond_name}']].mean(axis=1)
            print(cond_name)
            print(df_wavelet.describe())
            print(df_wavelet.sem())
            df_wavelet.dropna(inplace=True)  # Only want this for image
            no_kept = len(df_wavelet.index)
            # print(df_wavelet)

            fig, ax = plt.subplots(1, figsize=(16, 10))
            df_wavelet = pd.melt(df_wavelet, id_vars=['Subject'], value_vars=[f'Spinal', f'Subcortical', f'Cortical'],
                              var_name='CNS Level', value_name='Wavelets Count')  # Change to long format

            sns.pointplot(data=df_wavelet,
                        x='CNS Level', y='Wavelets Count', hue='Subject', palette=sns.color_palette(['gray'], no_kept),
                        ax=ax, legend=False)
            plt.setp(ax.collections, alpha=.1)  # for the markers
            plt.setp(ax.lines, alpha=.1)  # for the lines
            plt.legend([], [], frameon=False)
            sns.boxplot(data=df_wavelet,
                        x='CNS Level', y='Wavelets Count',
                        ax=ax, fill=False, palette=sns.color_palette(['tab:blue', 'tab:cyan', 'tab:purple']),
                        hue='CNS Level', legend=False, linewidth=3)
            plt.title(f"{cond_name}, {sheetname}")
            plt.savefig(figure_path + f'CrossCNS_PeaksTroughs_{cond_name}_{sheetname}.png')
            plt.savefig(
                figure_path + f'CrossCNS_PeaksTroughs_{cond_name}_{sheetname}.pdf',
                bbox_inches='tight', format="pdf")
            plt.close()