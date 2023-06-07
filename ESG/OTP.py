# Perform Oversampled Temporal Projection

import os
import mne
import pandas as pd
from Common_Functions.get_conditioninfo import *


def apply_OTP(subject, condition, srmr_nr, sampling_rate):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    if srmr_nr == 1:
        load_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
        save_path = "/data/pt_02718/tmp_data_otp/otp_cleaned/" + subject_id + "/"
        os.makedirs(save_path, exist_ok=True)
    elif srmr_nr == 2:
        load_path = "/data/pt_02718/tmp_data_2/imported/" + subject_id + "/"
        save_path = "/data/pt_02718/tmp_data_2_otp/otp_cleaned/" + subject_id + "/"
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
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'

    raw = mne.io.read_raw_fif(load_path + fname, preload=True)

    ##########################################################################################
    # OTP
    ##########################################################################################
    clean_raw = mne.preprocessing.oversampled_temporal_projection(raw)

    ##############################################################################################
    # Save
    ##############################################################################################
    # Save the SSP cleaned data for future comparison
    clean_raw.save(f"{save_path}otp_cleaned_{cond_name}.fif", fmt='double', overwrite=True)