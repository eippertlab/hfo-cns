# Plotting how the number of peaks and troughs varies across levels of the CNS for each subject


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        excel_fname = '/data/pt_02718/tmp_data/Peaks_Troughs_EqualWindow.xlsx'
        figure_path = f'/data/p_02718/Polished/Peak_Trough_Images_EqualWindow/CrossCNS_Change/'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        excel_fname = '/data/pt_02718/tmp_data_2/Peaks_Troughs_EqualWindow.xlsx'
        figure_path = '/data/p_02718/Polished_2/Peak_Trough_Images_EqualWindow/CrossCNS_Change/'
    os.makedirs(figure_path, exist_ok=True)
    data_types = ['spinal', 'subcortical', 'cortical']

    for cond_name in cond_names:
        df_merged = pd.DataFrame()
        for sheetname in ['10%', '20%', '25%', '33%', '50%']:
            df_pt = pd.read_excel(excel_fname, sheetname)
            col_names_troughs = [f'spinal_troughs_{cond_name}', f'subcortical_troughs_{cond_name}', f'cortical_troughs_{cond_name}']
            col_names_peaks = [f'spinal_peaks_{cond_name}', f'subcortical_peaks_{cond_name}', f'cortical_peaks_{cond_name}']

            # Need to get average of peaks and troughs for each subject and then make new df with these values
            for data_type in data_types:
                df = pd.DataFrame()
                df['Subject'] = subjects
                df['CNS Level'] = data_type
                df['Wavelets'] = df_pt[[f'{data_type}_peaks_{cond_name}', f'{data_type}_troughs_{cond_name}']].mean(axis=1)
                df['Threshold'] = sheetname
                df_merged = pd.concat([df_merged, df], axis=0, ignore_index=True)

        print(df_merged)
        # All on same plot
        fig, ax = plt.subplots(1, figsize=(16, 10))
        plt.legend([], [], frameon=False)
        sns.boxplot(x=df_merged['Threshold'], y=df_merged['Wavelets'], hue=df_merged['CNS Level'],
                    ax=ax, fill=False, palette=sns.color_palette(['tab:blue', 'tab:cyan', 'tab:purple']),
                    legend=True, linewidth=3, gap=0.1)
        plt.title(f"{cond_name}")
        if srmr_nr == 1:
            ax.set_ylim([1.4, 10])
        else:
            ax.set_ylim([0, 10])
        plt.savefig(figure_path + f'CrossCNS_CombinedThreshold_PeaksTroughs_{cond_name}_{sheetname}.png')
        plt.savefig(
            figure_path + f'CrossCNS_CombinedThreshold_PeaksTroughs_{cond_name}_{sheetname}.pdf',
            bbox_inches='tight', format="pdf")
        plt.close()

        # All on same plot
        fig, ax = plt.subplots(1, figsize=(16, 10))
        plt.legend([], [], frameon=False)
        sns.boxplot(x=df_merged['CNS Level'], y=df_merged['Wavelets'], hue=df_merged['Threshold'],
                    ax=ax, fill=False, legend=True, linewidth=3, gap=0.1)
        plt.title(f"{cond_name}")
        plt.savefig(figure_path + f'CrossCNS_MergedPeaksTroughs_{cond_name}_{sheetname}.png')
        plt.savefig(
            figure_path + f'CrossCNS_MergedPeaksTroughs_{cond_name}_{sheetname}.pdf',
            bbox_inches='tight', format="pdf")
        plt.close()