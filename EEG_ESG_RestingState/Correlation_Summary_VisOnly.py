########################################################################################################
# Read in the correlation matrices for each subject we previously generated
# Get the average correlation per subject and condition
# Save to excel table
########################################################################################################

import numpy as np
import pandas as pd
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', None)  # or 199

if __name__ == '__main__':
    freq_band = 'sigma'
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
    else:
        raise RuntimeError('Only implemented for srmr_nr 1')

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    excel_path = f'/data/pt_02718/{folder}/'
    excel_fname = f'{excel_path}Correlation_Long.xlsx'
    excel_sheetname = 'Correlation'
    col_names = ['Subject', 'cortical_median_task', 'cortical_median_rest',
                 'cortical_tibial_task', 'cortical_tibial_rest',
                 'subcortical_median_task', 'subcortical_median_rest',
                 'subcortical_tibial_task', 'subcortical_tibial_rest',
                 'spinal_median_task', 'spinal_median_rest',
                 'spinal_tibial_task', 'spinal_tibial_rest']
    df_corr = pd.read_excel(excel_fname, excel_sheetname)
    df_corr.set_index('Subject', inplace=True)

    df_vis_spinal = pd.read_excel('/data/pt_02718/tmp_data/Visibility_Updated.xlsx', 'CCA_Spinal')

    df_vis_subcortical = pd.read_excel('/data/pt_02718/tmp_data/Visibility_Thalamic_Updated.xlsx', 'CCA_Brain')

    df_vis_cortical = pd.read_excel('/data/pt_02718/tmp_data/Visibility_Updated.xlsx', 'CCA_Brain')

    for data_type in ['spinal', 'subcortical', 'cortical']:
        if data_type == 'spinal':
            df_vis = df_vis_spinal
        elif data_type == 'subcortical':
            df_vis = df_vis_subcortical
        else:
            df_vis = df_vis_cortical

        for condition in conditions:
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            for subject in subjects:
                vis = df_vis.loc[subject-1, f'Sigma_{cond_name.capitalize()}_Visible']
                if vis == 'F':
                    df_corr.at[subject, f'{data_type}_{cond_name}_task'] = np.nan
                    df_corr.at[subject, f'{data_type}_{cond_name}_rest'] = np.nan

    print(df_corr)
    print(df_corr.describe())
