###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
###############################################################################################

import numpy as np
from Common_Functions.import_data import import_data
from ESG.SSP import apply_SSP
from Common_Functions.bad_channel_check import bad_channel_check
from Common_Functions.bad_trial_check import bad_trial_check
from Common_Functions.Create_Frequency_Bands import create_frequency_bands
from ESG.run_CCA_spinal import run_CCA
from ESG.rm_heart_artefact import rm_heart_artefact

if __name__ == '__main__':
    ######## 1. Import ############
    import_d = True  # Prep work

    ######### 2. Clean the heart artefact using SSP ########
    SSP_flag = True
    no_projections = 6

    ######## Clean the heart artefact using SSP ###########
    pca_removal = True

    ######## 3. Bad Channel Check #######
    check_channels = True

    ######### 4. Bad trial check #############
    check_trials = True

    ######### 5. Split into frequency bands #############
    split_bands_flag = True

    ######### 6. Run CCA on each frequency band ##########
    CCA_flag = True

    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    # subjects = [1]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 5000  # Frequency to downsample to from original of 10kHz
    # Interested in frequencies up to 1200Hz

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
                apply_SSP(subject, condition, srmr_nr, sampling_rate, no_projections)

    ###################################################
    # Remove heart artefact using PCA_OBS method
    ##################################################
    if pca_removal:
        for subject in subjects:
            for condition in conditions:
                rm_heart_artefact(subject, condition, srmr_nr, sampling_rate)
                # If pchip is true, uses data where stim artefact was removed by pchip

    ###################################################
    # Bad Channel Check
    ###################################################
    if check_channels:
        for subject in subjects:
            for condition in conditions:
                bad_channel_check(subject, condition, srmr_nr, sampling_rate, channel_type='esg')

    ###################################################
    # Bad Channel Check
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
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma', 'kappa']:
                    run_CCA(subject, condition, srmr_nr, freq_band)
