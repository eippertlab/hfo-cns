###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
###############################################################################################

import numpy as np
# from Common_Functions.Create_Frequency_Bands_General import create_frequency_bands
from Common_Functions.Create_Frequency_Bands_Ktest import create_frequency_bands
from Common_Functions.keep_good_trials import keep_good_trials
from ESG.run_CCA_spinal_good import run_CCA_good
from ESG.run_CCA_spinal import run_CCA

if __name__ == '__main__':

    ######### 5. Split into frequency bands #############
    split_bands_flag = False

    ######### 6. Run CCA on each frequency band ##########
    CCA_flag = False

    ####### Test retaining only good trials ####
    keep_good = False

    ###### Run CCA on good trials only #######
    CCA_good_flag = True

    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    # subjects = [1]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ###################################################
    # Split into frequency bands of interest
    # Checking just generalised high frequency - 400 to 1000Hz
    ###################################################
    if split_bands_flag:
        for subject in subjects:
            for condition in conditions:
                create_frequency_bands(subject, condition, srmr_nr, sampling_rate, channel_type='esg')

    ###################################################
    # Run CCA on Freq Bands
    ###################################################
    if CCA_flag:
        for subject in subjects:
            for condition in conditions:
                # for freq_band in ['general']:
                for freq_band in ['ktest']:
                    run_CCA(subject, condition, srmr_nr, freq_band)

    ###################################################
    # Keep Good
    ###################################################
    if keep_good:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma']:
                    keep_good_trials(subject, condition, srmr_nr, freq_band)

    ###################################################
    # Run CCA on Freq Bands - good trials only
    ###################################################
    if CCA_good_flag:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma']:
                    run_CCA_good(subject, condition, srmr_nr, freq_band)
