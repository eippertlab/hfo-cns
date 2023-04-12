###############################################################################################
# Emma Bailey, 03/04/2023
# Wrapper script for project to investigate high frequency oscillations in the human spinal cord
# For bipolar electrodes
###############################################################################################

import numpy as np
from Bipolar.ImportBipolar_CNAP import import_data
from Bipolar.ImportBipolar_SequentialFit import import_dataepochs


if __name__ == '__main__':
    ######## 1. Import ############
    import_d = False  # Prep work

    ######## 1. Import ############
    import_depochs = True  # Removes stim artefact with sequential fitting method - have epochs saved after this

    n_subjects = 36  # Number of subjects
    # subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    subjects = [5]
    srmr_nr = 1  # Experiment Number
    conditions = [2, 3]  # Conditions of interest
    sampling_rate = 5000  # Frequency to downsample to from original of 10kHz
    sampling_rate_og = 10000
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
                import_data(subject, condition, srmr_nr, sampling_rate_og, repair_stim_art=True)

    if import_depochs:
        for subject in subjects:
            for condition in conditions:
                import_dataepochs(subject, condition, srmr_nr, sampling_rate)
