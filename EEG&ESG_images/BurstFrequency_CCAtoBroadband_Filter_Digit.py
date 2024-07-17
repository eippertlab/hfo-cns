# Plotting some initial images about the relation between frequency in different subjects


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    high_freq = 800
    subjects = np.arange(1, 25)
    cond_names = ['med_digits']
    figure_path = f'/data/p_02718/Images_2/BurstFrequencyAnalysis_CCAtoBroadband_Filter_Digits/'
    os.makedirs(figure_path, exist_ok=True)
    data_types = ['Thalamic', 'Cortical']
    excel_fname = f'/data/pt_02718/tmp_data_2/Peak_Frequency_400_{high_freq}_ccabroadband_filter_digits.xlsx'

    for cond_name in cond_names:
        df_combination = pd.DataFrame()
        df_combination['Subject'] = subjects
        col_name = f'Peak_Frequency_{cond_name}'
        for data_type in data_types:
            sheetname = data_type
            df_freq = pd.read_excel(excel_fname, sheetname)
            df_combination[f"{data_type}"] = df_freq[col_name]

        print(cond_name)
        print(df_combination.describe())
        print(df_combination.sem())

        df_combination.dropna(inplace=True)
        df_combination = pd.melt(df_combination, id_vars=['Subject'],
                                 value_vars=[f'Thalamic', f'Cortical'],
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
                        palette=['tab:cyan', 'tab:purple'])
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