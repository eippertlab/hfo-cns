########################################################################################################
# Read in the correlation matrices for each subject we previously generated
# Get the average correlation per subject and condition
# Save to excel table
########################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.check_excel_exist_general import check_excel_exist_general

if __name__ == '__main__':
    freq_band = 'sigma'
    srmr_nr = 1
    type = 'cca'  # can be long, shorter or cca

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
    else:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    excel_path = f'/data/pt_02718/{folder}/'
    if type == 'long':
        excel_fname = f'{excel_path}Correlation_Long.xlsx'
    elif type == 'shorter':
        excel_fname = f'{excel_path}Correlation_Shorter.xlsx'
    elif type == 'cca':
        excel_fname = f'{excel_path}Correlation_CCAWin.xlsx'
    excel_sheetname = 'Correlation'
    col_names = ['Subject', 'cortical_median_task', 'cortical_median_rest',
                 'cortical_tibial_task', 'cortical_tibial_rest',
                 'subcortical_median_task', 'subcortical_median_rest',
                 'subcortical_tibial_task', 'subcortical_tibial_rest',
                 'spinal_median_task', 'spinal_median_rest',
                 'spinal_tibial_task', 'spinal_tibial_rest']
    check_excel_exist_general(subjects, fname=excel_fname, sheetname=excel_sheetname, col_names=col_names)
    df_corr = pd.read_excel(excel_fname, excel_sheetname)
    df_corr.set_index('Subject', inplace=True)

    for data_type in ['spinal', 'subcortical', 'cortical']:
        for condition in conditions:
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            for subject in subjects:
                # Set variables
                subject_id = f'sub-{str(subject).zfill(3)}'
                eeg_chans, spin_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                if cond_name in ['median', 'med_mixed']:
                    esg_chans = cervical_chans
                else:
                    esg_chans = lumbar_chans

                # Select the right files
                if data_type == 'spinal':
                    input_path = f"/data/pt_02718/tmp_data/cca_rs/{subject_id}/"
                elif data_type == 'cortical':
                    input_path = f"/data/pt_02718/tmp_data/cca_eeg_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    input_path = f"/data/pt_02718/tmp_data/cca_eeg_thalamic_rs/{subject_id}/"

                if type == 'long':
                    fname_trig = f"{data_type}_corr_task_{cond_name}.pkl"
                    fname_rest = f"{data_type}_corr_rs_{cond_name}.pkl"
                elif type == 'shorter':
                    fname_trig = f"{data_type}_corr_task_shorter_{cond_name}.pkl"
                    fname_rest = f"{data_type}_corr_rs_shorter_{cond_name}.pkl"
                elif type == 'cca':
                    fname_trig = f"{data_type}_corr_task_ccawin_{cond_name}.pkl"
                    fname_rest = f"{data_type}_corr_rs_ccawin_{cond_name}.pkl"

                # All evoked objects run from time -0.1s up to 0.299s
                # Have shape n_trials, n_times
                # Need to crop to the time windows above
                # Our times are in steps of 0.0002 s (5000Hz sample rate)
                # These are lists of numpy.ndarray
                with open(f'{input_path}{fname_trig}', 'rb') as f:
                    corr_trig = pickle.load(f)

                with open(f'{input_path}{fname_rest}', 'rb') as f:
                    corr_rest = pickle.load(f)

                average_trig = corr_trig.mean(axis=(0, 1))
                average_rest = corr_rest.mean(axis=(0, 1))

                df_corr.at[subject, f'{data_type}_{cond_name}_task'] = average_trig
                df_corr.at[subject, f'{data_type}_{cond_name}_rest'] = average_rest

    # Write the dataframe to the excel file
    with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
        df_corr.to_excel(writer, sheet_name=excel_sheetname, columns=col_names[1:])