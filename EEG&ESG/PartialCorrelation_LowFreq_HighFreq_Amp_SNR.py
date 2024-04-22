# Looking now at partial correlation metrics
# Seeing if the relationships between high freq and low freq values exists even when SNR is accounted for


import numpy as np
import pandas as pd
import matplotlib as mpl
from scipy.stats import pearsonr
from pingouin import partial_corr
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        cond_names = ['median', 'tibial']
        excel_fname = f'/data/pt_02718/tmp_data/LowFreq_HighFreq_Amp_SNR.xlsx'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        cond_names = ['med_mixed', 'tib_mixed']
        excel_fname = f'/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Amp_SNR.xlsx'

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
            col_names_low_highenv = [f'{base}_amplitude', f'{base}_high_amplitude_env']
            col_names_low_high = [f'{base}_amplitude', f'{base}_high_amplitude']

            print('\n\n')
            print(f'{data_type}, {cond_name}')
            # Need to subselect only the bits associated with col_names of interest, otherwise for median_spinal
            # We lose subjects just because of their poor tibial performance
            relevant_cols = [col for col in df.columns if f'{base}' in col]
            df_rel = df.copy()[relevant_cols]
            df_rel.dropna(inplace=True)
            for col_names in [col_names_low_highenv, col_names_low_high]:
                pearson_corr = pearsonr(df_rel.abs()[f'{col_names[0]}'], df_rel.abs()[f'{col_names[1]}'])
                if col_names == col_names_low_highenv:
                    print('\n')
                    print('High frequency peak of envelope considered')
                else:
                    print('\n')
                    print('High frequency peak magnitude considered')
                print(pearson_corr)
                for covar_name in [f'{base}_SNR', f'{base}_high_SNR', [f'{base}_SNR', f'{base}_high_SNR']]:
                    print(f'Partial Correlation, controlling variable(s): {covar_name}')
                    stats = partial_corr(data=df_rel.abs(), x=f'{col_names[0]}', y=f'{col_names[1]}',
                                         covar=covar_name)
                    print(stats)
            # print(df[col_names_low_highenv].abs().corr('pearson'))  # Works but doesn't give a p-value
