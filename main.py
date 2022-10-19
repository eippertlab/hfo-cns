###############################################################################################
# Emma Bailey, 18/10/2022
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
###############################################################################################

import numpy as np
from import_data import import_data
from SSP import apply_SSP

if __name__ == '__main__':
    ######## Import ############
    import_d = True  # Prep work

    ######### Want to clean the heart artefact using SSP? ########
    SSP_flag = False  # Heart artefact removal by SSP
    no_projections = 6

    n_subjects = 36  # Number of subjects
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    # subjects = [1]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 5000  # Frequency to downsample to from original of 10kHz
    # Interested in frequencies up to 12000Hz

    ############################################
    # Import Data from BIDS directory
    # Select channels to analyse
    # Remove stimulus artefact if needed
    # Downsample and concatenate blocks of the same conditions
    # Detect QRS events
    # Save the new data and the QRS events
    ############################################
    if import_d:
        for subject in subjects:
            for condition in conditions:
                import_data(subject, condition, srmr_nr, sampling_rate)

    ##################################################
    # To remove heart artifact using SSP method in MNE
    # Also notch filters from 48Hz to 52Hz
    ###################################################
    if SSP_flag:
        for subject in subjects:
            for condition in conditions:
                apply_SSP(subject, condition, srmr_nr, sampling_rate, no_projections)
