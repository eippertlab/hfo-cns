# Plotting some initial images about the relation between frequency in different subjects


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    figure_to_plot = 2  # [1:splithalf, 2:across CNS]
    srmr_nr = 1
    high_freq = 800

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        figure_path = f'/data/p_02718/Images/BurstFrequencyAnalysis_CCAtoBroadband_Filter/'
        os.makedirs(figure_path, exist_ok=True)
        data_types = ['Spinal', 'Thalamic', 'Cortical']
        excel_fname = f'/data/pt_02718/tmp_data/Peak_Frequency_400_{high_freq}_ccabroadband_filter.xlsx'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        figure_path = f'/data/p_02718/Images_2/BurstFrequencyAnalysis_CCAtoBroadband_Filter/'
        os.makedirs(figure_path, exist_ok=True)
        data_types = ['Spinal', 'Thalamic', 'Cortical']
        excel_fname = f'/data/pt_02718/tmp_data_2/Peak_Frequency_400_{high_freq}_ccabroadband_filter.xlsx'

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
                plt.savefig(figure_path +f'SplitHalf_{data_type}_{cond_name}_{high_freq}.png')
                # plt.show()
                plt.close()

    elif figure_to_plot == 2:
        for cond_name in cond_names:
            df_combination = pd.DataFrame()
            df_combination['Subject'] = subjects
            col_name = f'Peak_Frequency_{cond_name}'
            for data_type in data_types:
                sheetname = data_type
                df_freq = pd.read_excel(excel_fname, sheetname)
                df_combination[f"{data_type}"] = df_freq[col_name]

            for index, row in df_combination.iterrows():
                if (row['Cortical'] >= row['Thalamic']) and (row['Thalamic'] >= row['Spinal']):
                    df_combination.at[index, 'Expected Pattern'] = True
                else:
                    df_combination.at[index, 'Expected Pattern'] = False
            print(cond_name)
            print(df_combination.describe())
            print(df_combination.sem())

            df_combination.dropna(inplace=True)
            df_combination = pd.melt(df_combination, id_vars=['Subject'],
                                     value_vars=[f'Spinal', f'Thalamic', f'Cortical'],
                                     var_name='CNS Level', value_name='Frequency')  # Change to long format

            g = sns.catplot(kind='point', data=df_combination, x='CNS Level', y='Frequency', hue='Subject', palette='dark:gray')
            for ax in g.axes.flat:
                for line in ax.lines:
                    line.set_alpha(0.3)
                for dots in ax.collections:
                    color = dots.get_facecolor()
                    dots.set_color(sns.set_hls_values(color, l=0.5))
                    dots.set_alpha(0.3)
            g.map_dataframe(sns.boxplot, x="CNS Level", y="Frequency", hue="CNS Level", dodge=False,
                            palette=['tab:blue', 'tab:cyan', 'tab:purple'])
            g.fig.set_size_inches(16, 10)
            g._legend.remove()
            plt.title(f"{cond_name}")
            plt.xlabel('CNS Level')
            plt.ylabel('Frequency (Hz)')
            ax.set_ylim([450, 750])
            plt.savefig(figure_path + f'CrossCNS_{cond_name}_{high_freq}.png')
            plt.savefig(
                figure_path + f'CrossCNS_{cond_name}_{high_freq}.pdf',
                bbox_inches='tight', format="pdf")
            # plt.show()
            plt.close()

    else:
        print('Error: Must specify plot 1 or 2')