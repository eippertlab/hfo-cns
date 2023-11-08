# We have all of the freuqencies computed already
# Read in these values and then remove those who don't have a valid CCA component chosen in the finish on either the
# spinal level, thalamic level OR cortical level  - - if they fail ANY of these levels, remove them

# Then recompute the average burst frequency at each level and save

import numpy as np
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo

if __name__ == '__main__':
    srmr_nr = 1
    fsearch_low = 400
    fsearch_high = 800  # 800 or 1200
    freq_band = 'Sigma'

    if srmr_nr == 1:
        burst_frequency_path = f'/data/pt_02718/tmp_data/Peak_Frequency_{fsearch_low}_{fsearch_high}.xlsx'
        thalamic_visibility_path = f'/data/pt_02718/tmp_data/Visibility_Thalamic_Updated.xlsx'
        visibility_path = f'/data/pt_02718/tmp_data/Visibility_Updated.xlsx'
        subjects = np.arange(1, 37)
        indices = np.arange(0, 36)
        conditions = [2, 3]

    elif srmr_nr == 2:
        burst_frequency_path = f'/data/pt_02718/tmp_data_2/Peak_Frequency_{fsearch_low}_{fsearch_high}.xlsx'
        thalamic_visibility_path = f'/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated.xlsx'
        visibility_path = f'/data/pt_02718/tmp_data_2/Visibility_Updated.xlsx'
        subjects = np.arange(1, 25)
        indices = np.arange(0, 24)
        conditions = [3, 5]

    df_overall = pd.DataFrame()

    for condition in conditions:
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        df_freq_spin = pd.read_excel(burst_frequency_path, 'Spinal')
        df_freq_spin.set_index('Subject', inplace=True)

        df_freq_thal = pd.read_excel(burst_frequency_path, 'Thalamic')
        df_freq_thal.set_index('Subject', inplace=True)

        df_freq_cort = pd.read_excel(burst_frequency_path, 'Cortical')
        df_freq_cort.set_index('Subject', inplace=True)

        df_spin_vis = pd.read_excel(visibility_path, 'CCA_Spinal')
        df_spin_vis.set_index('Subject', inplace=True)

        df_thal_vis= pd.read_excel(thalamic_visibility_path, 'CCA_Brain')
        df_thal_vis.set_index('Subject', inplace=True)

        df_cort_vis = pd.read_excel(visibility_path, 'CCA_Brain')
        df_cort_vis.set_index('Subject', inplace=True)

        spin_vis = df_spin_vis[f'{freq_band}_{cond_name.capitalize()}_Visible'].tolist()
        thal_vis = df_thal_vis[f'{freq_band}_{cond_name.capitalize()}_Visible'].tolist()
        cort_vis = df_cort_vis[f'{freq_band}_{cond_name.capitalize()}_Visible'].tolist()

        spin_removal = []
        thal_removal = []
        cort_removal = []
        if 'F' in spin_vis:
            spin_removal = [index for (index, item) in enumerate(spin_vis) if item == "F"]
        if 'F' in thal_vis:
            thal_removal = [index for (index, item) in enumerate(thal_vis) if item == "F"]
        if 'F' in cort_vis:
            cort_removal = [index for (index, item) in enumerate(cort_vis) if item == "F"]
        removal = set(spin_removal + thal_removal + cort_removal)

        keepers = list(set(indices) - removal)
        keepers.sort()

        for column_base in ['Peak_Frequency', 'Peak_Frequency_1', 'Peak_Frequency_2']:
            values_spin = df_freq_spin[f"{column_base}_{cond_name}"].tolist()
            values_thal = df_freq_thal[f"{column_base}_{cond_name}"].tolist()
            values_cort = df_freq_cort[f"{column_base}_{cond_name}"].tolist()

            kept_spin = np.mean([values_spin[i] for i in keepers])
            kept_thal = np.mean([values_thal[i] for i in keepers])
            kept_cort = np.mean([values_cort[i] for i in keepers])

            print(f'{column_base}_{cond_name}')
            print(kept_spin)
            print(kept_thal)
            print(kept_cort)
