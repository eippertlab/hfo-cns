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
from scipy.stats import pearsonr
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':

    srmr_nr = 2
    sfreq = 10000
    freq_band = 'sigma'

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        figure_path = f'/data/p_02718/Images/Bipolar_Images/LowFreq_HighFreq_LatencyRelation/'
        excel_fname = f'/data/pt_02718/tmp_data/Bipolar_Latency.xlsx'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        figure_path = f'/data/p_02718/Images_2/Bipolar_Images/LowFreq_HighFreq_LatencyRelation/'
        excel_fname = f'/data/pt_02718/tmp_data_2/Bipolar_Latency.xlsx'

    os.makedirs(figure_path, exist_ok=True)

    df_med = pd.DataFrame()
    df_tib = pd.DataFrame()
    df_med['Subject'] = subjects
    df_tib['Subject'] = subjects

    for cond_name in cond_names:
        # Get low freq data and high freq data and compile in separate dataframes
        for sheetname in ['LowFrequency', 'HighFrequency']:
            df = pd.read_excel(excel_fname, sheetname)

            if sheetname == 'LowFrequency':
                if cond_name in ['median', 'med_mixed']:
                    df_med[f'median_low'] = df[f'lat_{cond_name}']
                else:
                    df_tib[f'tibial_low'] = df[f'lat_{cond_name}']
            elif sheetname == 'HighFrequency':
                if cond_name in ['median', 'med_mixed']:
                    df_med[f'median_high'] = df[f'central_lat_{cond_name}']
                else:
                    df_tib[f'tibial_high'] = df[f'central_lat_{cond_name}']

    # Now relevant information is extracted, plot
    col_med = ['Subject', f'median_low', f'median_high']
    col_tib = ['Subject', f'tibial_low', f'tibial_high']

    for df, col_names in zip([df_med, df_tib],
                             [col_med, col_tib]):
        plt.figure()
        df = df[col_names]
        df.dropna(inplace=True)
        print(len(df[col_names]))
        pearson_corr = pearsonr(df.abs()[f'{col_names[1]}'], df.abs()[f'{col_names[2]}'])
        sns.scatterplot(data=df, x=col_names[1], y=col_names[2])
        if col_names == col_med:
            plt.title(f"Median, Low vs. High Frequency, PearsonCorrelation: {round(pearson_corr.statistic, 4)}, pval: {round(pearson_corr.pvalue, 4)}")
            fname = 'median_lowfreq'
        elif col_names == col_tib:
            plt.title(f"Tibial, Low vs. High Frequency, PearsonCorrelation: {round(pearson_corr.statistic, 4)}, pval: {round(pearson_corr.pvalue, 4)}")
            fname = 'tibial_highfreq'

        plt.savefig(figure_path + fname + 'scatter.png')
        plt.close()
