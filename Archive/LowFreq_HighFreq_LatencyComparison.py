# Plotting the latency differences
# Normal high: Latency of peak of HFO amplitude envelope
# Alternative high: Latency is middle point between places with 1/2 peak amplitude
# Want to compare low freq median/tibial spinal to low freq median/tibial cortical
# Want to compare high freq median/tibial spinal to high freq median/tibial cortical

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# import pingouin as pg
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':

    srmr_nr = 2
    sfreq = 5000
    freq_band = 'sigma'
    alternative_flag = True

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        figure_path = f'/data/p_02718/Images/LowFreq_HighFreq_LatencyRelation/'
        if alternative_flag:
            raise RuntimeError('Alternative is no longer in use, change flag and restart')
            # excel_fname = f'/data/pt_02718/tmp_data/LowFreq_HighFreq_RelationAlternative.xlsx'
        else:
            excel_fname = f'/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        figure_path = f'/data/p_02718/Images_2/LowFreq_HighFreq_LatencyRelation/'

        if alternative_flag:
            excel_fname = f'/data/pt_02718/tmp_data_2/LowFreq_HighFreq_RelationAlternative.xlsx'
        else:
            excel_fname = f'/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx'

    os.makedirs(figure_path, exist_ok=True)

    df_med = pd.DataFrame()
    df_tib = pd.DataFrame()
    df_med['Subject'] = subjects
    df_tib['Subject'] = subjects

    for cond_name in cond_names:
        # Get low freq data and high freq data and compile in separate dataframes
        for sheetname in ['Spinal', 'Cortical']:
            df = pd.read_excel(excel_fname, sheetname)
            if sheetname == 'Spinal':
                if cond_name in ['median', 'med_mixed']:
                    base = 'N13'
                else:
                    base = 'N22'
            elif sheetname == 'Cortical':
                if cond_name in ['median', 'med_mixed']:
                    base = 'N20'
                else:
                    base = 'P39'

            if cond_name in ['median', 'med_mixed']:
                df_med[f'{base}'] = df[f'{base}']
                df_med[f'{base}_high'] = df[f'{base}_high']
            else:
                df_tib[f'{base}'] = df[f'{base}']
                df_tib[f'{base}_high'] = df[f'{base}_high']

    # Now relevant information is extracted, plot
    col_med = ['Subject', f'N13', f'N20']
    col_med_high = ['Subject', f'N13_high', f'N20_high']
    col_tib = ['Subject', f'N22', f'P39']
    col_tib_high = ['Subject', f'N22_high', f'P39_high']

    for df, col_names in zip([df_med, df_tib, df_med, df_tib],
                             [col_med, col_tib, col_med_high, col_tib_high]):
        plt.figure()
        df = df[col_names]
        df_combination = pd.melt(df, id_vars=['Subject'],
                                 value_vars=[col_names[1], col_names[2]],
                                 var_name='Potential', value_name='Latency')  # Change to long format

        g = sns.catplot(kind='point', data=df_combination, x='Potential', y='Latency', hue='Subject')
        g.fig.set_size_inches(8, 8)
        if col_names == col_med:
            plt.title(f"Median, Low Frequency")
            fname = 'median_lowfreq'
        elif col_names == col_med_high:
            plt.title(f"Median, High Frequency")
            fname = 'median_highfreq'
        elif col_names == col_tib:
            plt.title(f"Tibial, Low Frequency")
            fname = 'tibial_lowfreq'
        elif col_names == col_tib_high:
            plt.title(f"Tibial, High Frequency")
            fname = 'tibial_highfreq'

        if alternative_flag:
            plt.savefig(figure_path + fname + '_alternative.png')
            plt.close()
        else:
            plt.savefig(figure_path + fname + '.png')
            plt.close()

    for df, col_names in zip([df_med, df_tib, df_med, df_tib],
                             [col_med, col_tib, col_med_high, col_tib_high]):
        plt.figure()
        df = df[col_names]
        sns.scatterplot(data=df, x=col_names[1], y=col_names[2])
        if col_names == col_med:
            plt.title(f"Median, Low Frequency")
            fname = 'median_lowfreq'
        elif col_names == col_med_high:
            plt.title(f"Median, High Frequency")
            fname = 'median_highfreq'
        elif col_names == col_tib:
            plt.title(f"Tibial, Low Frequency")
            fname = 'tibial_lowfreq'
        elif col_names == col_tib_high:
            plt.title(f"Tibial, High Frequency")
            fname = 'tibial_highfreq'

        if alternative_flag:
            plt.savefig(figure_path + fname + 'scatter_alternative.png')
            plt.close()
        else:
            plt.savefig(figure_path + fname + 'scatter.png')
            plt.close()

