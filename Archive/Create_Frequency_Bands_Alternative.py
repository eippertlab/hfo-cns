# Read in the data and split into 3 frequency bands
# Save each of the 3 bands as a separate raw signals

import os
import mne
import pandas as pd
from Common_Functions.get_conditioninfo import *


def create_frequency_bands(subject, condition, srmr_nr, sampling_rate, channel_type, both_patches):
    subject_id = f'sub-{str(subject).zfill(3)}'

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve
    nblocks = cond_info.nblocks

    if channel_type == 'esg':
        # set variables
        if srmr_nr == 1:
            load_path = "/data/pt_02718/tmp_data_otp/ssp_cleaned/" + subject_id + "/"
            save_path = "/data/pt_02718/tmp_data_otp/freq_banded_esg/" + subject_id + "/"
            os.makedirs(save_path, exist_ok=True)
            if both_patches:
                fname = f'ssp6_cleaned_{cond_name}.fif'
            else:
                fname = f'ssp6_cleaned_{cond_name}_separatepatch.fif'
        elif srmr_nr == 2:
            load_path = "/data/pt_02718/tmp_data_2_otp/ssp_cleaned/" + subject_id + "/"
            save_path = "/data/pt_02718/tmp_data_2_otp/freq_banded_esg/" + subject_id + "/"
            os.makedirs(save_path, exist_ok=True)
            fname = f'ssp6_cleaned_{cond_name}.fif'

    elif channel_type == 'eeg':
        # set variables
        if srmr_nr == 1:
            load_path = "/data/pt_02718/tmp_data_otp/otp_cleaned_eeg/" + subject_id + "/"
            fname = f'otp_cleaned_{cond_name}.fif'
            save_path = "/data/pt_02718/tmp_data_otp/freq_banded_eeg/" + subject_id + "/"
        elif srmr_nr == 2:
            load_path = "/data/pt_02718/tmp_data_2_otp/otp_cleaned_eeg/" + subject_id + "/"
            fname = f'otp_cleaned_{cond_name}.fif'
            save_path = "/data/pt_02718/tmp_data_2_otp/freq_banded_eeg/" + subject_id + "/"

    os.makedirs(save_path, exist_ok=True)
    raw = mne.io.read_raw_fif(load_path + fname, preload=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    sigma = [df.loc[df['var_name'] == 'sigma_low', 'var_value'].iloc[0],
             df.loc[df['var_name'] == 'sigma_high', 'var_value'].iloc[0]]
    kappa = [df.loc[df['var_name'] == 'k_low', 'var_value'].iloc[0],
             df.loc[df['var_name'] == 'k_high', 'var_value'].iloc[0]]
    # kappa1 = [df.loc[df['var_name'] == 'k1_low', 'var_value'].iloc[0],
    #           df.loc[df['var_name'] == 'k1_high', 'var_value'].iloc[0]]
    # kappa2 = [df.loc[df['var_name'] == 'k2_low', 'var_value'].iloc[0],
    #           df.loc[df['var_name'] == 'k2_high', 'var_value'].iloc[0]]

    # band_dict = {'sigma': sigma,
    #              'kappa1': kappa1,
    #              'kappa2': kappa2}

    # band_dict = {'sigma': sigma,
    #              'kappa': kappa}

    band_dict = {'sigma': sigma}
    for band_name in band_dict.keys():
        raw.filter(l_freq=band_dict[band_name][0], h_freq=band_dict[band_name][1], n_jobs=len(raw.ch_names), method='iir',
                   iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')
        if channel_type == 'esg' and both_patches is False:
            raw.save(f"{save_path}{band_name}_{cond_name}_separatepatch.fif", fmt='double', overwrite=True)
        else:
            raw.save(f"{save_path}{band_name}_{cond_name}.fif", fmt='double', overwrite=True)

