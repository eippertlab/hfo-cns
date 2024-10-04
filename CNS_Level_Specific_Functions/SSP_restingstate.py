# Perform Signal Space Projection
# Mainly working from tutorial https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html
# SSP uses singular value decomposition to create the projection matrix

import os
import mne
import pandas as pd
from Common_Functions.get_conditioninfo import *


def apply_SSP_restingstate(subject, condition, srmr_nr, sampling_rate, n_p):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    if srmr_nr == 1:
        load_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
        save_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
        cond_names_trig = ['median', 'tibial']
        os.makedirs(save_path, exist_ok=True)
    elif srmr_nr == 2:
        load_path = "/data/pt_02718/tmp_data_2/imported/" + subject_id + "/"
        save_path = "/data/pt_02718/tmp_data_2/ssp_cleaned/" + subject_id + "/"
        cond_names_trig = ['med_mixed', 'tib_mixed']
        os.makedirs(save_path, exist_ok=True)
    else:
        print('Error: Experiment 1 or 2 must be specified')

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name

    for cond_name_trig in cond_names_trig:
        ###########################################################################################
        # Load
        ###########################################################################################
        # load imported ESG data
        fname = f'noStimart_sr{sampling_rate}_{cond_name}_{cond_name_trig}_withqrs.fif'

        raw = mne.io.read_raw_fif(load_path + fname, preload=True)

        ##########################################################################################
        # SSP
        ##########################################################################################
        projs, events = mne.preprocessing.compute_proj_ecg(raw, n_eeg=n_p, reject=None,
                                                           n_jobs=len(raw.ch_names), ch_name='ECG')

        # Apply projections (clean data)
        clean_raw = raw.copy().add_proj(projs)
        clean_raw = clean_raw.apply_proj()

        ##############################################################################################
        # Save
        ##############################################################################################
        # Save the SSP cleaned data for future comparison
        clean_raw.save(f"{save_path}ssp{n_p}_cleaned_{cond_name}_{cond_name_trig}.fif", fmt='double', overwrite=True)

        # To save space - delete imported file after we have the SSP cleaned version
        if os.path.exists(load_path+fname):
            os.remove(load_path+fname)
