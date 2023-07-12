# Perform Signal Space Projection
# Mainly working from tutorial https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html
# SSP uses singular value decomposition to create the projection matrix

import os
import mne
import pandas as pd
from Common_Functions.get_conditioninfo import *


def apply_SSP(subject, condition, srmr_nr, sampling_rate, n_p, both_patches):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    if srmr_nr == 1:
        load_path = "/data/pt_02718/tmp_data_otp/otp_cleaned/" + subject_id + "/"
        save_path = "/data/pt_02718/tmp_data_otp/ssp_cleaned/" + subject_id + "/"
        os.makedirs(save_path, exist_ok=True)
    elif srmr_nr == 2:
        load_path = "/data/pt_02718/tmp_data_2_otp/otp_cleaned/" + subject_id + "/"
        save_path = "/data/pt_02718/tmp_data_2_otp/ssp_cleaned/" + subject_id + "/"
        os.makedirs(save_path, exist_ok=True)
    else:
        print('Error: Experiment 1 or 2 must be specified')

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve
    nblocks = cond_info.nblocks

    ###########################################################################################
    # Load
    ###########################################################################################
    # load imported ESG data
    if both_patches:
        fname = f"otp_cleaned_{cond_name}.fif"
    else:
        fname = f"otp_cleaned_{cond_name}_separatepatch.fif"

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
    if both_patches:
        clean_raw.save(f"{save_path}ssp{n_p}_cleaned_{cond_name}.fif", fmt='double', overwrite=True)
    else:
        clean_raw.save(f"{save_path}ssp{n_p}_cleaned_{cond_name}_separatepatch.fif", fmt='double', overwrite=True)

