# Want to extract frequency of oscillation vs whether they later had a chosen CCA component
# Looking at both spinal, thalamic and cortical
# Mixed nerve condition for both dataset 1 and 2

import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.calculate_snr_hfo import calculate_snr
from Common_Functions.calculate_snr_lowfreq import calculate_snr_lowfreq
import pandas as pd
import matplotlib as mpl
from Common_Functions.check_excel_exist_general import check_excel_exist_general
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    ##############################################################################################################
    # Set paths and variables
    ##############################################################################################################
    data_types = ['Spinal', 'Thalamic', 'Cortical']

    sfreq = 5000
    freq_band = 'sigma'

    for srmr_nr in [1, 2]:
        if srmr_nr == 1:
            subjects = np.arange(1, 37)
            conditions = [2, 3]
            folder = 'tmp_data'
        elif srmr_nr == 2:
            subjects = np.arange(1, 25)
            conditions = [3, 5]
            folder = 'tmp_data_2'

        # Cortical Excel file
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA_Brain')
        df_cortical.set_index('Subject', inplace=True)

        # Thalamic Excel file
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Thalamic_Updated.xlsx')
        df_thal = pd.read_excel(xls, 'CCA_Brain')
        df_thal.set_index('Subject', inplace=True)

        # Spinal Excel file
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Updated.xlsx')
        df_spinal = pd.read_excel(xls, 'CCA_Spinal')
        df_spinal.set_index('Subject', inplace=True)

        # Frequency Excel file
        xls = f'/data/pt_02718/{folder}/Peak_Frequency_400_1200.xlsx'

        excel_fname = f'/data/pt_02718/{folder}/FreqVsSelection.xlsx'

        for data_type in data_types:
            if data_type == 'Spinal':
                df_vis = df_spinal
            elif data_type == 'Thalamic':
                df_vis = df_thal
            elif data_type == 'Cortical':
                df_vis = df_cortical

            for condition in conditions:  # Conditions (median, tibial) or (med_mixed, tib_mixed)
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name

                # Read in freq excel
                df_freq = pd.read_excel(xls, f'{data_type}')
                df_freq.set_index('Subject', inplace=True)

                # Dataframe for saving
                sheetname = f'{data_type}_{cond_name}'
                check_excel_exist_general(subjects, excel_fname, sheetname, col_names=['Subject', 'frequency', 'component_selected'])
                df_save = pd.read_excel(excel_fname, sheetname)
                df_save.set_index('Subject', inplace=True)

                for subject in subjects:  # All subjects
                    df_save.at[subject, f'frequency'] = df_freq.loc[subject, f'Peak_Frequency_{cond_name}']
                    df_save.at[subject, f'component_selected'] = df_vis.loc[subject, f'Sigma_{cond_name.capitalize()}_Visible']

                # Write the dataframe to the excel file
                with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
                    df_save.to_excel(writer, sheet_name=sheetname)