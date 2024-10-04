###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
###############################################################################################
# Can run a bad channel check for data quality purposes, but no channels are excluded before running CCA

import numpy as np
from Common_Functions.import_data_rs import import_data
from CNS_Level_Specific_Functions.SSP_restingstate import apply_SSP_restingstate
from Common_Functions.Create_Frequency_Bands_RS import create_frequency_bands_rs


if __name__ == '__main__':
    srmr_nr = 2  # Set the experiment number

    if srmr_nr == 1:
        n_subjects = 36  # Number of subjects
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [1]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    elif srmr_nr == 2:
        n_subjects = 24  # Number of subjects
        subjects = np.arange(1, 25)
        conditions = [1]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ######## 1. Import ############
    import_d = True  # Prep work

    ######### 2. Clean the heart artefact using SSP ########
    SSP_flag = True
    no_projections = 6

    ######### 3. Split into frequency bands #############
    split_bands_flag = True

    ############################################
    # Import Data from BIDS directory
    # Select channels to analyse
    # Remove stimulus artefact by PCHIP interpolation
    # Downsample and concatenate blocks of the same conditions
    # Also notch filters powerline noise and hpf at 1Hz
    ############################################
    if import_d:
        for subject in subjects:
            for condition in conditions:
                import_data(subject, condition, srmr_nr, sampling_rate, esg_flag=True)

    ##################################################
    # To remove heart artifact using SSP method in MNE
    ###################################################
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP_restingstate(subject, condition, srmr_nr, sampling_rate, no_projections)

    ###################################################
    # Split into frequency bands of interest
    ###################################################
    if split_bands_flag:
        for subject in subjects:
            for condition in conditions:
                create_frequency_bands_rs(subject, condition, srmr_nr, sampling_rate, channel_type='esg')