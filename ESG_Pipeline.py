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
from Common_Functions.keep_good_trials import keep_good_trials
from ESG.run_CCA_spinal_good import run_CCA_good
from Archive.run_CCA_spinal_opposite import run_CCA_oppo
from Archive.run_CCA_spinal_opposite_anterior import run_CCA_oppo_anterior
from Archive.rm_heart_artefact import rm_heart_artefact

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
        # subjects = np.arange(1, 25)   # 1 through 36 to access subject data
        subjects = np.arange(1, 2)
        conditions = [2, 3, 4, 5]  # Conditions of interest - tib digits and med digits, also including mixed nerve now
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ######## 1. Import ############
    import_d = False  # Prep work

    ######### 2. Clean the heart artefact using SSP ########
    SSP_flag = False
    no_projections = 6

    ######## 3. Bad Channel Check #######
    check_channels = False

    ######### 4. Bad trial check #############
    check_trials = False

    ######### 5. Split into frequency bands #############
    split_bands_flag = False

    ######### 6. Run CCA on each frequency band ##########
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
                import_data(subject, condition, srmr_nr, sampling_rate, esg_flag=True)

    ##################################################
    # To remove heart artifact using SSP method in MNE
    ###################################################
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP(subject, condition, srmr_nr, sampling_rate, no_projections)

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
    # Old: Run CCA on Freq Bands
    ###################################################
    if CCA_flag:
        for subject in subjects:
            for condition in conditions:
                for freq_band in ['sigma', 'kappa']:
                    run_CCA(subject, condition, srmr_nr, freq_band)

    ###################################################################################################################
    # GRAVEYARD
    ###################################################################################################################
    # ######### 6. Keep only the good trials ###########
    # keep_good = False
    #
    # ######## 7. Run CCA on only the good trials #########
    # CCA_good_flag = False
    #
    # ###################################################
    # # Keep Good
    # ###################################################
    # if keep_good:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma']:
    #                 keep_good_trials(subject, condition, srmr_nr, freq_band, 'esg')
    #
    # ###################################################
    # # Run CCA on Freq Bands - good trials only
    # ###################################################
    # if CCA_good_flag:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma']:
    #                 run_CCA_good(subject, condition, srmr_nr, freq_band)

    # ###################################################
    # # Remove heart artefact using PCA_OBS method
    # ##################################################
    # ######## Clean the heart artefact using SSP ###########
    # pca_removal = False
    # if pca_removal:
    #     for subject in subjects:
    #         for condition in conditions:
    #             rm_heart_artefact(subject, condition, srmr_nr, sampling_rate)
    #             # If pchip is true, uses data where stim artefact was removed by pchip

    # ######### Extra. Run CCA on opposite patch - control analyses #########
    # CCA_oppo_flag = False
    #
    # ######### Extra. Run CCA on opposite patch - control analyses #########
    # CCA_oppo_ant_flag = False

    # ###################################################
    # # Run CCA on the opposite patch
    # # To check for spatial specificity
    # ###################################################
    # if CCA_oppo_flag:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma']:
    #                 run_CCA_oppo(subject, condition, srmr_nr, freq_band)
    #
    # ###################################################
    # # Run CCA on the opposite patch, data anteriorly rereferenced
    # # To check for spatial specificity
    # ###################################################
    # if CCA_oppo_ant_flag:
    #     for subject in subjects:
    #         for condition in conditions:
    #             for freq_band in ['sigma']:
    #                 run_CCA_oppo_anterior(subject, condition, srmr_nr, freq_band)

