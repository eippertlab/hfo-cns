# Take the previously computed SNR values for each component, fold and subject
# Compute overall values instead by averaging across folds for each subject and component


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    srmr_nr = 2
    mode = 'Spinal'  # Can be Brain, Thalamic or Spinal
    kfolds = 5
    n_components = 4

    if srmr_nr == 1:
        app_folder = ""
    elif srmr_nr == 2:
        app_folder = "_2"

    if mode == 'Brain':
        excel_fname = f'/data/pt_02718/tmp_data{app_folder}/SNR_Peak_{kfolds}fold_EEG_Updated.xlsx'
    elif mode == 'Thalamic':
        excel_fname = f'/data/pt_02718/tmp_data{app_folder}/SNR_Peak_{kfolds}fold_EEG_Thalamic_Updated.xlsx'
    elif mode == 'Spinal':
        excel_fname = f'/data/pt_02718/tmp_data{app_folder}/SNR_Peak_{kfolds}fold_Updated.xlsx'
    else:
        raise ValueError('Mode must be selected as either Brain, Thalamic or Spinal')

    df = pd.read_excel(excel_fname, sheet_name='SNR_Peak')
    df_avg = pd.DataFrame()

    # Want average of just the SNR columns across the folds for each component
    for comp in range(n_components):
        if srmr_nr == 1:
            median_cols = [f"sigma_median_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                           for
                           option in ['SNR']]
            tibial_cols = [f"sigma_tibial_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                           for
                           option in ['SNR']]
        elif srmr_nr == 2:
            median_cols = [f"sigma_med_mixed_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                           for
                           option in ['SNR']]
            tibial_cols = [f"sigma_tib_mixed_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                           for
                           option in ['SNR']]
        df_avg[f'med_comp{comp+1}'] = df[median_cols].mean(axis=1)
        df_avg[f'tib_comp{comp+1}'] = df[tibial_cols].mean(axis=1)
    print(df_avg)
    print(f'{mode}_{kfolds}folds_AverageSNRAcrossParticipants')
    print(df_avg.mean())

    # Now looking at the SNR values if only subjects with T for the Peak being in correct zone are included
    df_avg_reduced = pd.DataFrame()
    for comp in range(n_components):
        if srmr_nr == 1:
            med_name = 'median'
            tib_name = 'tibial'
        elif srmr_nr == 2:
            med_name = 'med_mixed'
            tib_name = 'tib_mixed'
        median_cols = [f"sigma_{med_name}_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                       for
                       option in ['SNR']]
        tibial_cols = [f"sigma_{tib_name}_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                       for
                       option in ['SNR']]
        median_cols_peak = [f"sigma_{med_name}_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                       for
                       option in ['Peak']]
        tibial_cols_peak = [f"sigma_{tib_name}_fold{x + 1}_comp{comp + 1}_{option}" for x in range(kfolds)
                       for
                       option in ['Peak']]

        df_current_median = df[median_cols+median_cols_peak].copy()
        df_current_tibial = df[tibial_cols+tibial_cols_peak].copy()

        # Keep only rows where ALL Peak values are T across that row
        median_mask = df_current_median[median_cols_peak].eq('T').all(axis=1)
        tibial_mask = df_current_tibial[tibial_cols_peak].eq('T').all(axis=1)

        # Average remaining SNR values across the row
        df_current_median.loc[median_mask, f"sigma_{med_name}_comp{comp + 1}_SNR_avg"] = df_current_median.loc[median_mask, median_cols].mean(axis=1)
        df_current_tibial.loc[tibial_mask, f"sigma_{tib_name}_comp{comp + 1}_SNR_avg"] = df_current_tibial.loc[tibial_mask, tibial_cols].mean(axis=1)

        df_avg_reduced[f'med_comp{comp + 1}'] = df_current_median[f"sigma_{med_name}_comp{comp + 1}_SNR_avg"]
        df_avg_reduced[f'tib_comp{comp + 1}'] = df_current_tibial[f"sigma_{tib_name}_comp{comp + 1}_SNR_avg"]

    print(df_avg_reduced)
    print(f'{mode}_{kfolds}folds_AverageSNRAcrossParticipants')
    print(df_avg_reduced.mean(skipna=True))
    print(df_avg_reduced.count())


