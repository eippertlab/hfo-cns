###############################################################################################
# Emma Bailey, 24/05/2023
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
###############################################################################################

import numpy as np
from Common_Functions.import_data import import_data
from ESG.OTP import apply_OTP
from ESG.SSP_Alternative import apply_SSP
from Common_Functions.bad_trial_check_Alternative import bad_trial_check
from Common_Functions.Create_Frequency_Bands_Alternative import create_frequency_bands
from ESG.run_CCA_spinal_Alternative import run_CCA
from ESG.run_CCA_spinal_2_Alternative import run_CCA2

if __name__ == '__main__':
    srmr_nr = 1  # Set the experiment number

    if srmr_nr == 1:
        n_subjects = 36  # Number of subjects
        # subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        subjects = [6]  # [15, 18, 25, 26]  # First 2 I currently reject median, second 2 tibial
        conditions = [2, 3]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    elif srmr_nr == 2:
        n_subjects = 24  # Number of subjects
        subjects = np.arange(1, 25)
        conditions = [2, 3, 4, 5]  # Conditions of interest - tib digits and med digits, also including mixed nerve now
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ######## 1. Import ############
    import_d = False  # Prep work

    ######### 2. Run OTP #########
    otp_flag = True

    ######### 3. Clean the heart artefact using SSP ########
    SSP_flag = True
    no_projections = 6

    ######### 4. Bad trial check #############
    check_trials = True

    ######### 5. Split into frequency bands #############
    split_bands_flag = True

    ######### 6. Run CCA on each frequency band ##########
    CCA_flag = True

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
    # To remove uncorrelated sensor noise
    ###################################################
    if otp_flag:
        for subject in subjects:
            for condition in conditions:
                apply_OTP(subject, condition, srmr_nr, sampling_rate)

    ##################################################
    # To remove heart artifact using SSP method in MNE
    ###################################################
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP(subject, condition, srmr_nr, sampling_rate, no_projections)

    ###################################################
    # Bad Trial Check
    ###################################################
    if check_trials:
        for subject in subjects:
            for condition in conditions:
                bad_trial_check(subject, condition, srmr_nr, sampling_rate, channel_type='esg')

    ###################################################
    # Split into frequency bands of interest
    ###################################################
    if split_bands_flag:
        for subject in subjects:
            for condition in conditions:
                create_frequency_bands(subject, condition, srmr_nr, sampling_rate, channel_type='esg')

    ###################################################
    # Run CCA on Freq Bands
    ###################################################
    if CCA_flag:
        if srmr_nr == 1:
            for subject in subjects:
                for condition in conditions:
                    for freq_band in ['sigma']:
                        run_CCA(subject, condition, srmr_nr, freq_band)
        elif srmr_nr == 2:
            conditions_d2 = [2, 4]  # only need to specify digits, takes care of mixed nerve within other script
            for subject in subjects:
                for condition in conditions_d2:
                    for freq_band in ['sigma']:
                        run_CCA2(subject, condition, srmr_nr, freq_band)
