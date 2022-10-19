# Perform Signal Space Projection
# Mainly working from tutorial https://mne.tools/stable/auto_tutorials/preprocessing/50_artifact_correction_ssp.html
# SSP uses singular value decomposition to create the projection matrix

import os
import mne
import pandas as pd
from get_conditioninfo import *


def apply_SSP(subject, condition, srmr_nr, sampling_rate, n_p):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    load_path = "/data/pt_02718/imported/" + subject_id + "/"
    save_path = "/data/pt_02718/ssp/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve
    nblocks = cond_info.nblocks

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    notch_low = df.loc[df['var_name'] == 'notch_freq_low', 'var_value'].iloc[0]
    notch_high = df.loc[df['var_name'] == 'notch_freq_high', 'var_value'].iloc[0]

    ###########################################################################################
    # Load
    ###########################################################################################
    # load imported ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'

    raw = mne.io.read_raw_fif(load_path + fname, preload=True)

    ##########################################################################################
    # SSP
    ##########################################################################################
    projs, events = mne.preprocessing.compute_proj_ecg(raw, n_eeg=n_p, reject=None, n_jobs=len(raw.ch_names),
                                                       ch_name='ECG')

    # Apply projections (clean data)
    clean_raw = raw.copy().add_proj(projs)
    clean_raw = clean_raw.apply_proj()

    ##############################################################################################
    # Reference and Remove Powerline Noise
    ##############################################################################################
    # make sure recording reference is included
    mne.add_reference_channels(clean_raw, ref_channels=['TH6'], copy=False)  # Modifying in place

    clean_raw.notch_filter(freqs=[notch_low, notch_high], n_jobs=len(raw.ch_names), method='fir', phase='zero')

    ##############################################################################################
    # Save
    ##############################################################################################
    # Save the SSP cleaned data for future comparison
    clean_raw.save(f"{save_path}ssp{n_p}_cleaned_{cond_name}.fif", fmt='double', overwrite=True)
