###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human brain
###############################################################################################
# Can run a bad channel check for data quality purposes, but no channels are excluded before running CCA

import numpy as np
from Common_Functions.import_data import import_data
from Common_Functions.bad_trial_check import bad_trial_check
from Common_Functions.bad_channel_check import bad_channel_check
from Common_Functions.automated_blinkremoval import run_icablinkremoval
from Common_Functions.Create_Frequency_Bands import create_frequency_bands
from EEG.run_CCA_brain import run_CCA
from EEG.run_CCA_brain_2 import run_CCA2
from EEG.run_CCA_brain_thalamic import run_CCA_thalamic
from EEG.run_CCA_brain_thalamic_2 import run_CCA_thalamic2

if __name__ == '__main__':
    srmr_nr = 1  # Set the experiment number

    if srmr_nr == 1:
        n_subjects = 36  # Number of subjects
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [2, 3]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    elif srmr_nr == 2:
        n_subjects = 24  # Number of subjects
        subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
        conditions = [2, 3, 4, 5]  # Conditions of interest - tib digits and med digits, also including mixed nerve now
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ######## 1. Import ############
    import_d = False  # Prep work

    ######## 2. Bad Channel Check ###########
    check_channels = False  # No longer drop channels before CCA is run

    ######## 3. Bad Trial Check ###########
    check_trials = False

    ######## 4. Freq band ##########
    split_bands_flag = False

    # ######## 4. a) Automated eyeblink correction #####
    # remove_eyeblinks = False

    ######## 5. Run CCA on cortical activity ########
    freq_type = 'low'
    CCA_flag = True

    ######## 6. Run CCA on subcortical activity  ########
    CCA_thalamic_flag = False

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

    # ###################################################
    # # Remove eyeblinks
    # ###################################################
    # if remove_eyeblinks:
    #     for subject in subjects:
    #         for condition in conditions:
    #             run_icablinkremoval(subject, condition, srmr_nr, sampling_rate)

    ###################################################
    # Run CCA on Freq Bands - all trials
    ###################################################
    if CCA_flag:
        if srmr_nr == 1:
            for subject in subjects:
                for condition in conditions:
                    for freq_band in ['sigma']:
                        run_CCA(subject, condition, srmr_nr, freq_band, sampling_rate, freq_type)
        elif srmr_nr == 2:
            conditions_d2 = [2, 4]  # only need to specify digits, takes care of mixed nerve within other script
            for subject in subjects:
                for condition in conditions_d2:
                    for freq_band in ['sigma']:
                        run_CCA2(subject, condition, srmr_nr, freq_band, sampling_rate, freq_type)

    ###################################################
    # Run CCA on subcortical activity - all trials
    ###################################################
    if CCA_thalamic_flag:
        if srmr_nr == 1:
            for subject in subjects:
                for condition in conditions:
                    for freq_band in ['sigma']:
                        run_CCA_thalamic(subject, condition, srmr_nr, freq_band, sampling_rate, freq_type)
        elif srmr_nr == 2:
            conditions_d2 = [2, 4]  # only need to specify digits, takes care of mixed nerve within other script
            for subject in subjects:
                for condition in conditions_d2:
                    for freq_band in ['sigma']:
                        run_CCA_thalamic2(subject, condition, srmr_nr, freq_band, sampling_rate, freq_type)