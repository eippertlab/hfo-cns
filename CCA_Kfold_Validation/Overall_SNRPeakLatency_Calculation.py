# Take the previously computed SNR and peak latency values for each component, fold and subject
# Get the average of the snr and latency columns
# Looking at only component 1
# Check SNR is above threshold and that latency is within signal window


import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    srmr_nr = 1
    mode = 'Spinal'  # Can be Brain, Thalamic or Spinal
    kfolds = 10
    n_components = 4

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    if srmr_nr == 1:
        if mode == 'Brain':
            excel_fname = f'/data/pt_02718/tmp_data/SNR_PeakLatency_{kfolds}fold_EEG_Updated.xlsx'
            sep_latency_med = df_timing.loc[df_timing['Name'] == 'centre_cort_med', 'Time'].iloc[0] / 1000
            signal_window_med = df_timing.loc[df_timing['Name'] == 'edge_cort_med', 'Time'].iloc[0] / 1000
            sep_latency_tib = df_timing.loc[df_timing['Name'] == 'centre_cort_tib', 'Time'].iloc[0] / 1000
            signal_window_tib = df_timing.loc[df_timing['Name'] == 'edge_cort_tib', 'Time'].iloc[0] / 1000
        elif mode == 'Thalamic':
            excel_fname = f'/data/pt_02718/tmp_data/SNR_PeakLatency_{kfolds}fold_EEG_Thalamic_Updated.xlsx'
            sep_latency_med = df_timing.loc[df_timing['Name'] == 'centre_sub_med', 'Time'].iloc[0] / 1000
            signal_window_med = df_timing.loc[df_timing['Name'] == 'edge_sub_med', 'Time'].iloc[0] / 1000
            sep_latency_tib = df_timing.loc[df_timing['Name'] == 'centre_sub_tib', 'Time'].iloc[0] / 1000
            signal_window_tib = df_timing.loc[df_timing['Name'] == 'edge_sub_tib', 'Time'].iloc[0] / 1000
        elif mode == 'Spinal':
            excel_fname = f'/data/pt_02718/tmp_data/SNR_PeakLatency_{kfolds}fold_Updated.xlsx'
            sep_latency_med = df_timing.loc[df_timing['Name'] == 'centre_spinal_med', 'Time'].iloc[0] / 1000
            signal_window_med = df_timing.loc[df_timing['Name'] == 'edge_spinal_med', 'Time'].iloc[0] / 1000
            sep_latency_tib = df_timing.loc[df_timing['Name'] == 'centre_spinal_tib', 'Time'].iloc[0] / 1000
            signal_window_tib = df_timing.loc[df_timing['Name'] == 'edge_spinal_tib', 'Time'].iloc[0] / 1000
        else:
            raise ValueError('Mode must be selected as either Brain, Thalamic or Spinal')

    else:
        raise RuntimeError('srmr_nr=2 is not yet implemented')

    df = pd.read_excel(excel_fname, sheet_name='SNR_Peak')
    df_avg_med = pd.DataFrame()
    df_avg_tib = pd.DataFrame()

    # Want average of just the SNR columns across the folds for each component
    median_cols = [f"sigma_median_fold{x + 1}_comp1_{option}" for x in range(kfolds)
                   for option in ['Peak']]
    tibial_cols = [f"sigma_tibial_fold{x + 1}_comp1_{option}" for x in range(kfolds)
                   for option in ['Peak']]
    median_cols_snr = [f"sigma_median_fold{x + 1}_comp1_{option}" for x in range(kfolds)
                   for option in ['SNR']]
    tibial_cols_snr = [f"sigma_tibial_fold{x + 1}_comp1_{option}" for x in range(kfolds)
                   for option in ['SNR']]
    df_avg_med[f'med_comp1_snr'] = df[median_cols_snr].mean(axis=1)
    df_avg_med[f'med_comp1_latency'] = df[median_cols].mean(axis=1)
    df_avg_tib[f'tib_comp1_snr'] = df[tibial_cols_snr].mean(axis=1)
    df_avg_tib[f'tib_comp1_latency'] = df[tibial_cols].mean(axis=1)

    # Check conditions are met
    if kfolds == 5:
        snr_min = 2.24
    elif kfolds == 10:
        snr_min = 1.58
    for cond_name, df_avg in zip(['med', 'tib'], [df_avg_med, df_avg_tib]):
        if cond_name == 'med':
            latency_min = sep_latency_med - signal_window_med
            latency_max = sep_latency_med + signal_window_med
        elif cond_name == 'tib':
            latency_min = sep_latency_tib - signal_window_tib
            latency_max = sep_latency_tib + signal_window_tib

        df_avg['meets_conditions'] = (
                (df_avg[f'{cond_name}_comp1_snr'] > snr_min) &
                (df_avg[f'{cond_name}_comp1_latency'] >= latency_min) &
                (df_avg[f'{cond_name}_comp1_latency'] <= latency_max)
        ).map({True: 'T', False: 'F'})
        t_count = (df_avg['meets_conditions'] == 'T').sum()
        print(f"{cond_name}, {mode}, {kfolds}folds: {t_count}")


