# Perform McNemars test
# Looking at whether having a frequency over 1000Hz affects the likelihood of CCA being able to find a meaningful
# component as per our criteria

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from Common_Functions.get_conditioninfo import get_conditioninfo

if __name__ == '__main__':
    srmr_nr = 2
    data_types = ['Spinal', 'Thalamic', 'Cortical']

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [3, 2]
        excel_path = '/data/pt_02718/tmp_data/'
        excel_fname = 'FreqVsSelection.xlsx'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        excel_path = '/data/pt_02718/tmp_data_2/'
        excel_fname = 'FreqVsSelection.xlsx'

    for data_type in data_types:
        for condition in conditions:
            # Set variables
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name

            xls = pd.ExcelFile(excel_path + excel_fname)
            df = pd.read_excel(xls, f'{data_type}_{cond_name}')
            df.set_index('Subject', inplace=True)

            # Need an above 1000HZ and below 1000Hz as wider headings
            # Within that we have the count of true and false
            df_contingency = pd.DataFrame()
            df_contingency.at['Selected_Component', 'Below_1000Hz'] = len(
                df[(df['frequency'] <= 1000) & (df['component_selected'] == 'T')])

            df_contingency.at['Selected_Component', 'Above_1000Hz'] = len(
                df[(df['frequency']>1000) & (df['component_selected']=='T')])

            df_contingency.at['No_Selected_Component', 'Below_1000Hz'] = len(
                df[(df['frequency'] <= 1000) & (df['component_selected'] == 'F')])

            df_contingency.at['No_Selected_Component', 'Above_1000Hz'] = len(
                df[(df['frequency'] > 1000) & (df['component_selected'] == 'F')])

            data_contingency = [
                [len(df[(df['frequency'] <= 1000) & (df['component_selected'] == 'T')]),
                 len(df[(df['frequency']>1000) & (df['component_selected']=='T')])],
                [len(df[(df['frequency'] <= 1000) & (df['component_selected'] == 'F')]),
                 len(df[(df['frequency'] > 1000) & (df['component_selected'] == 'F')])]
            ]
            print(f'{data_type}, {cond_name}')
            print(df_contingency)
            # print(data_contingency)
            # exit()
            # exact = false means chi-squared distribution is used, otherwise binomial
            # continuity correction should be true if values in cells are less than 4 and chi square used
            # I don't think McNemar test makes sense in our context
            # print('McNemar')
            # print(mcnemar(data_contingency, exact=False, correction=False))
            # print(mcnemar(data_contingency, exact=True, correction=False))

            # Cell counts are too small for a chi-squared test
            # res = chi2_contingency(data_contingency)
            # print('Chi-Square')
            # print(f'pvalue      {res.pvalue}')
            # print(f'statistic   {res.statistic}')

            # Fisher test: p-value below significance level tells us there is a relationship between frequency and
            # component selection
            # Fisher test needs at least 1 non-zero value in each row or column
            # Additive smoothing
            if any(0.0 in sl for sl in data_contingency):
                data_smoothed = [[n+1 for n in data_list] for data_list in data_contingency]
                print(data_smoothed)
                res = fisher_exact(data_smoothed, alternative='two-sided')
            else:
                res = fisher_exact(data_contingency, alternative='two-sided')

            print('Fisher-Exact')
            print(f'pvalue      {res.pvalue}')
            print(f'statistic   {res.statistic}')
            print('\n')
