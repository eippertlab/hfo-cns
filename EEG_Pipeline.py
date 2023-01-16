###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human brain
###############################################################################################

import numpy as np
from Common_Functions.import_data import import_data
from Common_Functions.bad_trial_check import bad_trial_check
from Common_Functions.bad_channel_check import bad_channel_check
from Common_Functions.Create_Frequency_Bands import create_frequency_bands
from EEG.run_CCA_brain import run_CCA
from Archive.run_CSP_brain import run_CSP
from Archive.run_bCSTP_brain import run_bCSTP


if __name__ == '__main__':
    ######## 1. Import ############
    import_d = False  # Prep work

    ######## 2. Bad Channel Check ###########
    check_channels = False

    ######## 3. Bad Trial Check ###########
    check_trials = True

    ######## 4. Freq band ##########
    split_bands_flag = True

    ######## 5. Run CCA ########
    CCA_flag = True

    ######## 6. Run CSP ########
    CSP_flag = False

    ######## 7. Run bCSTP #######
    bCSTP_flag = False

    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    # subjects = [1]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

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
                import_data(subject, condition, srmr_nr, sampling_rate, esg_flag=False)

    ###################################################
    # Bad Channel Check
    ###################################################
    if check_channels:
        for subject in subjects:
            for condition in conditions:
                bad_channel_check(subject, condition, srmr_nr, sampling_rate, channel_type='eeg')

    ###################################################
    # Bad Trial Check
    ###################################################
    if check_trials:
        for subject in subjects:
            for condition in conditions:
                bad_trial_check(subject, condition, srmr_nr, sampling_rate, channel_type='eeg')

    ###################################################
    # Split into frequency bands of interest
    ###################################################
    if split_bands_flag:
        for subject in subjects:
            for condition in conditions:
                create_frequency_bands(subject, condition, srmr_nr, sampling_rate, channel_type='eeg')

    ###################################################
    # Run CCA on Freq Bands
    ###################################################
    if CCA_flag:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma', 'kappa']:
                    run_CCA(subject, condition, srmr_nr, freq_band, sampling_rate)

    ###################################################
    # Run CSP on Freq Bands
    ###################################################
    if CSP_flag:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma', 'kappa']:
                    run_CSP(subject, condition, srmr_nr, freq_band, sampling_rate)

    ###################################################
    # Run bCSTP on Freq Bands
    ###################################################
    if bCSTP_flag:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma', 'kappa']:
                    run_bCSTP(subject, condition, srmr_nr, freq_band, sampling_rate)


