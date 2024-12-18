# Plotting some initial images about the relation between high and low frequency activity


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy.stats import pearsonr
import os
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 1
    envelope = False  # If envelope is True, plut high freq env, otherwise the actual peak of the HFO time trace

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        figure_path = f'/data/p_02718/Images/LowFreq_HighFreq_Relation/'
        excel_fname = f'/data/pt_02718/tmp_data/LowFreq_HighFreq_Amp_SNR.xlsx'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        figure_path = f'/data/p_02718/Images_2/LowFreq_HighFreq_Relation/'
        excel_fname = f'/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Amp_SNR.xlsx'
    os.makedirs(figure_path, exist_ok=True)

    data_types = ['Spinal', 'Cortical']

    for cond_name in cond_names:
        for data_type in data_types:
            if data_type == 'Spinal':
                if cond_name in ['median', 'med_mixed']:
                    base = 'N13'
                else:
                    base = 'N22'
            elif data_type == 'Cortical':
                if cond_name in ['median', 'med_mixed']:
                    base = 'N20'
                else:
                    base = 'P39'

            sheetname = data_type
            df = pd.read_excel(excel_fname, sheetname)
            if envelope:
                df_latency = df[['Subject', f'{base}', f'{base}_high_env']]
                col_lat = [f'{base}', f'{base}_high_env']
                df_amplitude = df[['Subject', f'{base}_amplitude', f'{base}_high_amplitude_env']]
                col_amp = [f'{base}_amplitude', f'{base}_high_amplitude_env']
            else:
                df_latency = df[['Subject', f'{base}', f'{base}_high']]
                col_lat = [f'{base}', f'{base}_high']
                df_amplitude = df[['Subject', f'{base}_amplitude', f'{base}_high_amplitude']]
                col_amp = [f'{base}_amplitude', f'{base}_high_amplitude']
            # var_name no holds half type, value_name holds the actual peak freq

            for df, col_names in zip([df_latency, df_amplitude], [col_lat, col_amp]):
                # df = pd.melt(df.abs(), id_vars=['Subject'], value_vars=col_names,
                #                   var_name='Potential', value_name='Value')  # Change to long format
                sns.scatterplot(data=df.abs(),
                                    x=col_names[0], y=col_names[1])
                # pearson_corr_og = df.abs()[f'{col_names[0]}'].corr(df.abs()[f'{col_names[1]}'])
                df.dropna(inplace=True)
                pearson_corr = pearsonr(df.abs()[f'{col_names[0]}'], df.abs()[f'{col_names[1]}'])
                # g.fig.set_size_inches(16, 10)
                plt.title(f"{data_type}, {cond_name}, PearsonCorrelation: {round(pearson_corr.statistic, 4)}, pval: {round(pearson_corr.pvalue, 4)}")
                if envelope:
                    plt.savefig(figure_path + f'{col_names[0]}_{data_type}_{cond_name}_abs_env.png')
                    plt.savefig(figure_path + f'{col_names[0]}_{data_type}_{cond_name}_abs_env.pdf',
                                    bbox_inches='tight', format="pdf")
                else:
                    plt.savefig(figure_path + f'{col_names[0]}_{data_type}_{cond_name}_abs.png')
                    plt.savefig(figure_path + f'{col_names[0]}_{data_type}_{cond_name}_abs.pdf',
                                bbox_inches='tight', format="pdf")
                # plt.show()
                plt.close()