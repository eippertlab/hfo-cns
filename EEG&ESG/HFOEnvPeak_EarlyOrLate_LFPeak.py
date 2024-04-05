# Want to count how many times the HFO amplitude envelope peak is before versus after the low frequency potential peak


import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    data_types = ['Spinal', 'Cortical']  # Not implemented for Thalamic - difficulties with LF-SEP

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    srmr_nr = 2
    hfo_type = 'actual'  # Can be 'env' - to use HFO envelope peak or 'actual' - to use biggest peak irrespective of polarity
    # of actual HFOs
    sfreq = 5000
    freq_band = 'sigma'

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]

    results_dict = {}
    print(hfo_type)
    for data_type in data_types:
        # Using Amp_SNR as we have computed peak of amp envelope and also peak of HFO time course within same bounds
        # Make sure our excel sheet is in place to store the values
        if srmr_nr == 1:
            excel_fname = f'/data/pt_02718/tmp_data/LowFreq_HighFreq_Amp_SNR.xlsx'
        elif srmr_nr == 2:
            excel_fname = f'/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Amp_SNR.xlsx'
        sheetname = data_type
        df_rel = pd.read_excel(excel_fname, sheetname)

        for condition in conditions:
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            results_dict[f'{data_type}_{cond_name}'] = {}
            for subject in subjects:  # All subjects
                hf_latency = np.nan
                lf_latency = np.nan

                subject_id = f'sub-{str(subject).zfill(3)}'

                if cond_name in ['tibial', 'tib_mixed']:
                    if data_type == 'Cortical':
                        pot_name = 'P39'
                    # elif data_type == 'Thalamic':
                    #     pot_name = 'P30'
                    elif data_type == 'Spinal':
                        pot_name = 'N22'

                elif cond_name in ['median', 'med_mixed']:
                    if data_type == 'Cortical':
                        pot_name = 'N20'
                    # elif data_type == 'Thalamic':
                    #     pot_name = 'P14'
                    elif data_type == 'Spinal':
                        pot_name = 'N13'

                lf_latency = df_rel.at[subject-1, pot_name]
                # Need to check amplitude since this excel has default time values if no actual time was found
                # This happens if no CCA comp selected or if there is no neg value for an N22 for example
                if pd.isna(df_rel.at[subject-1, f'{pot_name}_high_amplitude']):
                    hf_latency = np.nan
                else:
                    if hfo_type == 'actual':
                        hf_latency = df_rel.at[subject-1, f'{pot_name}_high']
                    elif hfo_type == 'env':
                        hf_latency = df_rel.at[subject-1, f'{pot_name}_high_env']

                if hf_latency > lf_latency:
                    results_dict[f'{data_type}_{cond_name}'][subject-1] = 'later'
                elif hf_latency <= lf_latency:
                    results_dict[f'{data_type}_{cond_name}'][subject - 1] = 'earlier'
                else:
                    results_dict[f'{data_type}_{cond_name}'][subject - 1] = np.nan

            print(f'{data_type}_{cond_name}')
            count_earl = 0
            count_late = 0
            count_nan = 0
            for key in results_dict[f'{data_type}_{cond_name}']:
                if results_dict[f'{data_type}_{cond_name}'][key] == 'later':
                    count_late += 1
                elif results_dict[f'{data_type}_{cond_name}'][key] == 'earlier':
                    count_earl += 1
                else:
                    count_nan += 1
            print(f'Earlier: {count_earl}')
            print(f'Later: {count_late}')
            print(f'NaN: {count_nan}')
