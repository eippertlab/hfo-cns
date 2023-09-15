###############################################################################################
# Emma Bailey, 30/08/2022
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
###############################################################################################
# This script only works if we rerun the bad_trial check after SSP to avoid filtering the raw we resave between
# 400 and 1400Hz

import numpy as np
from Archive.run_CCA_spinal_highlow import run_CCA_highlow
# from ESG.run_CCA_spinal_2 import run_CCA2_highlow

if __name__ == '__main__':
    srmr_nr = 1  # Set the experiment number

    if srmr_nr == 1:
        n_subjects = 36  # Number of subjects
        subjects = np.arange(1, 7)  # 1 through 36 to access subject data
        # subjects = [1]
        conditions = [2, 3]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    elif srmr_nr == 2:
        n_subjects = 24  # Number of subjects
        subjects = np.arange(1, 25)
        conditions = [2, 3, 4, 5]  # Conditions of interest - tib digits and med digits, also including mixed nerve now
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ######### 6. Run CCA on each frequency band ##########
    CCA_lowhigh_flag = True

    ###################################################
    # Run CCA on Freq Bands
    ###################################################
    if CCA_lowhigh_flag:
        if srmr_nr == 1:
            for subject in subjects:
                for condition in conditions:
                    for freq_band in ['sigma']:
                        run_CCA_highlow(subject, condition, srmr_nr, freq_band)
        # elif srmr_nr == 2:
        #     conditions_d2 = [2, 4]  # only need to specify digits, takes care of mixed nerve within other script
        #     for subject in subjects:
        #         for condition in conditions_d2:
        #             for freq_band in ['sigma']:
        #                 run_CCA2_highlow(subject, condition, srmr_nr, freq_band)