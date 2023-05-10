###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human brain
###############################################################################################

import numpy as np
from Common_Functions.import_data import import_data
from Common_Functions.bad_trial_check import bad_trial_check
from Common_Functions.bad_channel_check import bad_channel_check
from Common_Functions.Create_Frequency_Bands import create_frequency_bands
from Common_Functions.keep_good_trials import keep_good_trials
from Archive.run_CCA_brain_good import run_CCA_good
from EEG.run_CCA_brain import run_CCA

if __name__ == '__main__':
    srmr_nr = 2  # Set the experiment number

    if srmr_nr == 1:
        n_subjects = 36  # Number of subjects
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        # subjects = [1]
        conditions = [2, 3]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    elif srmr_nr == 2:
        n_subjects = 24  # Number of subjects
        # Testing with just subject 1 at the moment
        subjects = np.arange(1, 2)  # (1, 25) # 1 through 24 to access subject data
        conditions = [2, 3, 4, 5]  # Conditions of interest - tib digits and med digits, also including mixed nerve now
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ######## 1. Import ############
    import_d = True  # Prep work

    ######## 2. Bad Channel Check ###########
    check_channels = False

    ######## 3. Bad Trial Check ###########
    check_trials = False

    ######## 4. Freq band ##########
    split_bands_flag = False

    ######## 5. Run CCA ########
    CCA_flag = False

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
    # Run CCA on Freq Bands - all trials
    ###################################################
    if CCA_flag:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma']:
                    run_CCA(subject, condition, srmr_nr, freq_band, sampling_rate)

    ###################################################################################################################
    # GRAVEYARD
    ###################################################################################################################
    # ####### 6. Keep Good Trials #######
    # CCA_good_flag = False
    #
    # ###################################################
    # # Run CCA on Freq Bands - good trials only
    # ###################################################
    # if CCA_good_flag:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma']:
    #                 run_CCA_good(subject, condition, srmr_nr, freq_band, sampling_rate)


    # ####### 5. Keep Good Trials #######
    # keep_good = False
    ###################################################
    # Keep Good
    ###################################################
    # if keep_good:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma']:
    #                 keep_good_trials(subject, condition, srmr_nr, freq_band, 'eeg')
    #

    # ######## Old. Run CSP ########
    # CSP_flag = False
    #
    # ######## Old. Run bCSTP #######
    # bCSTP_flag = False
    # ###################################################
    # # Old: Run CSP on Freq Bands
    # ###################################################
    # if CSP_flag:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma', 'kappa']:
    #                 run_CSP(subject, condition, srmr_nr, freq_band, sampling_rate)
    #
    # ###################################################
    # # Old: Run bCSTP on Freq Bands
    # ###################################################
    # if bCSTP_flag:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma', 'kappa']:
    #                 run_bCSTP(subject, condition, srmr_nr, freq_band, sampling_rate)


