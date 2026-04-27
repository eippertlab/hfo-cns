"""
Wrapper script to run the CCA evoked data validation
Aim:
- Train CCA using half of the data
- Use the resulting CCA filter on the other half of the data
- Make the group level plots and compare to the full CCA model
"""

import numpy as np
from CCA_Kfold_Validation.run_CCA_spinal import run_CCA_spinal
from CCA_Kfold_Validation.run_CCA_brain import run_CCA_brain
from CCA_Kfold_Validation.run_CCA_brain_thalamic import run_CCA_thalamic


if __name__ == "__main__":
    srmr_nr = 2
    freq_type = 'high'
    k = 5

    if srmr_nr == 1:
        n_subjects = 36  # Number of subjects
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [2, 3]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    elif srmr_nr == 2:
        n_subjects = 24  # Number of subjects
        subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
        conditions = [3, 5]  # Conditions of interest
        sampling_rate = 5000  # Frequency to downsample to from original of 10kHz

    ###################################################
    # Run CCA model validation
    ####################################################
    # Brain
    for subject in subjects:
        for condition in conditions:
            for freq_band in ['sigma']:
                run_CCA_brain(subject, condition, srmr_nr, freq_band, sampling_rate, freq_type, k)

    # Subcortical
    for subject in subjects:
        for condition in conditions:
            for freq_band in ['sigma']:
                run_CCA_thalamic(subject, condition, srmr_nr, freq_band, sampling_rate, freq_type, k)

    # Spinal
    for subject in subjects:
        for condition in conditions:
            for freq_band in ['sigma']:
                run_CCA_spinal(subject, condition, srmr_nr, freq_band, freq_type, k)