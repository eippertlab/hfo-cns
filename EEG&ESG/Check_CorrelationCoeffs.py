# Check if correlation coeffs are different than 0 in shuffled data


import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from scipy.stats import ttest_1samp
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

def Extract(lst, pos):
    return [item[pos] for item in lst]

if __name__ == '__main__':
    cca_type = 'normal'  # shufflebyhalf
    data_types = ['eeg', 'esg']
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_band = 'sigma'
    srmr_nr = 1

    for data_type in data_types:
        for condition in conditions:
            correlations = []

            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Select the right files
                if cca_type == 'shufflebyhalf':
                    if data_type == 'esg':
                        input_path = f"/data/pt_02718/tmp_data/cca_shufflebyhalf/{subject_id}/"
                    elif data_type == 'eeg':
                        input_path = f"/data/pt_02718/tmp_data/cca_eeg_shufflebyhalf/{subject_id}/"
                elif cca_type == 'normal':
                    if data_type == 'esg':
                        input_path = f"/data/pt_02718/tmp_data/cca/{subject_id}/"
                    elif data_type == 'eeg':
                        input_path = f"/data/pt_02718/tmp_data/cca_eeg/{subject_id}/"

                ###########################################################
                # Spatial Pattern Extraction for HFOs
                ############################################################
                # Read in saved A_st
                with open(f'{input_path}r_{freq_band}_{cond_name}.pkl', 'rb') as f:
                    r = pickle.load(f)
                    correlations.append(list(r[0:4]))

            for position in [0, 1, 2, 3]:
                corr = Extract(correlations, position)
                # Testing null hypothesis that correlations are equal to 0
                # If p-value is less than 0.05, we reject the null hypothesis
                result_twosided = ttest_1samp(corr, 0)
                result_onesided = ttest_1samp(corr, 0, alternative='greater')
                print(f'For {data_type}, {cond_name}, component {position+1}, two-sided p-val is {result_twosided.pvalue},'
                      f' {result_twosided.confidence_interval(confidence_level=0.95)}')

