# Plotting some initial images about the relation between frequency in different subjects


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    figure_to_plot = 1  # [1:splithalf, 2:across CNS]
    before_CCA = False
    srmr_nr = 2
    fsearch_low = 400
    fsearch_high = 1200

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        figure_path = f'/data/p_02718/Images/BurstFrequencyAnalysis/'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        figure_path = f'/data/p_02718/Images_2/BurstFrequencyAnalysis/'
    os.makedirs(figure_path, exist_ok=True)
    data_types = ['Spinal', 'Thalamic', 'Cortical']

    if before_CCA:
        if srmr_nr == 1:
            excel_fname = f'/data/pt_02718/tmp_data/Peak_Frequency_{fsearch_low}_{fsearch_high}.xlsx'
        elif srmr_nr == 2:
            excel_fname = f'/data/pt_02718/tmp_data_2/Peak_Frequency_{fsearch_low}_{fsearch_high}.xlsx'
    else:
        if srmr_nr == 1:
            excel_fname = '/data/pt_02718/tmp_data/Peak_Frequency_CCA.xlsx'
        elif srmr_nr == 2:
            excel_fname = '/data/pt_02718/tmp_data_2/Peak_Frequency_CCA.xlsx'
        if figure_to_plot == 1:
            print('Error: Cannot do split-half with CCA data')
            exit()

    if figure_to_plot == 1:
        for cond_name in cond_names:
            col_names = [f'Peak_Frequency_1_{cond_name}', f'Peak_Frequency_2_{cond_name}']
            for data_type in data_types:
                sheetname = data_type
                df_freq = pd.read_excel(excel_fname, sheetname)
                df_freq = df_freq[['Subject', f'Peak_Frequency_1_{cond_name}', f'Peak_Frequency_2_{cond_name}']]
                # df_freq.set_index('Subject', inplace=True)
                # var_name no holds half type, value_name holds the actual peak freq
                df_freq = pd.melt(df_freq, id_vars=['Subject'], value_vars=col_names,
                                  var_name ='Timing', value_name ='Frequency')  # Change to long format

                g = sns.catplot(kind='point',
                            data=df_freq,
                            x='Timing', y='Frequency', hue='Subject')
                g.fig.set_size_inches(16, 10)
                plt.title(f"{data_type}, {cond_name}")
                if before_CCA:
                    plt.savefig(figure_path +f'SplitHalf_{data_type}_{cond_name}_{fsearch_high}.png')
                else:
                    plt.savefig(figure_path +f'SplitHalfAfterCCA_{data_type}_{cond_name}.png')
                # plt.show()
                plt.close()

    elif figure_to_plot == 2:
        df_combination = pd.DataFrame()
        df_combination['Subject'] = subjects
        for cond_name in cond_names:
            col_name = f'Peak_Frequency_{cond_name}'
            for data_type in data_types:
                sheetname = data_type
                df_freq = pd.read_excel(excel_fname, sheetname)
                df_combination[f"{data_type}_{col_name}"] = df_freq[col_name]

            df_combination = pd.melt(df_combination, id_vars=['Subject'], value_vars=[f'Spinal_{col_name}', f'Thalamic_{col_name}', f'Cortical_{col_name}'],
                              var_name='CNS Level', value_name='Frequency')  # Change to long format

            g = sns.catplot(kind='point',
                        data=df_combination,
                        x='CNS Level', y='Frequency', hue='Subject')
            g.fig.set_size_inches(16, 10)
            plt.title(f"{cond_name}")
            if before_CCA:
                plt.savefig(figure_path + f'CrossCNS_{cond_name}_{fsearch_high}.png')
            else:
                plt.savefig(figure_path + f'CrossCNSAfterCCA_{cond_name}.png')

            # plt.show()
            plt.close()

    else:
        print('Error: Must specify plot 1 or 2')