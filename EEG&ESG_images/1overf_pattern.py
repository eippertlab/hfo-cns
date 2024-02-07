import pandas as pd
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


if __name__ == '__main__':
    srmr_nr = 2
    freq_band = 'Sigma'

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        add_on = ''

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        add_on = '_2'

    excel_fname_coeff = f'/data/pt_02718/tmp_data{add_on}/1overf.xlsx'
    excel_fname_vis = f'/data/pt_02718/tmp_data{add_on}/Visibility_Updated.xlsx'

    for data_type in ['Spinal', 'Cortical']:
        sheetname_coeff = data_type
        if data_type == 'Spinal':
            sheetname_vis = 'CCA_Spinal'
        else:
            sheetname_vis = 'CCA_Brain'
        df_coeff = pd.read_excel(excel_fname_coeff, sheetname_coeff)
        df_coeff.set_index('Subject', inplace=True)

        df_vis = pd.read_excel(excel_fname_vis, sheetname_vis)
        df_vis.set_index('Subject', inplace=True)

        df = df_coeff.join(df_vis)
        df = df.iloc[:-1]
        print(df.describe())

        # Check change based on visible/not
        if srmr_nr == 1:
            df_med_vis = df[df[f'{freq_band}_Median_Visible'] == 'T']['median_exponent']
            df_med_not = df[df[f'{freq_band}_Median_Visible'] == 'F']['median_exponent']
            print('Median, Visible')
            print(df_med_vis.describe())
            print('Median, Not Visible')
            print(df_med_not.describe())
            df_tib_vis = df[df[f'{freq_band}_Tibial_Visible'] == 'T']['tibial_exponent']
            df_tib_not = df[df[f'{freq_band}_Tibial_Visible'] == 'F']['tibial_exponent']
            print('Tibial, Visible')
            print(df_tib_vis.describe())
            print('Tibial, Not Visible')
            print(df_tib_not.describe())
        elif srmr_nr == 2:
            df_med_vis = df[df[f'{freq_band}_Med_mixed_Visible'] == 'T']['med_mixed_exponent']
            df_med_not = df[df[f'{freq_band}_Med_mixed_Visible'] == 'F']['med_mixed_exponent']
            print('Median, Visible')
            print(df_med_vis.describe())
            print('Median, Not Visible')
            print(df_med_not.describe())
            df_tib_vis = df[df[f'{freq_band}_Tib_mixed_Visible'] == 'T']['tib_mixed_exponent']
            df_tib_not = df[df[f'{freq_band}_Tib_mixed_Visible'] == 'F']['tib_mixed_exponent']
            print('Tibial, Visible')
            print(df_tib_vis.describe())
            print('Tibial, Not Visible')
            print(df_tib_not.describe())

        # for condition in conditions:
        #     cond_info = get_conditioninfo(condition, srmr_nr)
        #     cond_name = cond_info.cond_name



