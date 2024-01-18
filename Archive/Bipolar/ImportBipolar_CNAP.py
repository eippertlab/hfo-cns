# Import necessary packages
import mne
from Common_Functions.get_conditioninfo import *
from Common_Functions.get_channels import *
from scipy.io import loadmat
import os
import glob
import numpy as np
import pandas as pd
from Common_Functions.pchip_interpolation import PCHIP_interpolation


# if __name__ == '__main__':
def import_data(subject, condition, srmr_nr, sampling_rate_og, repair_stim_art):
    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    save_path = "../tmp_data/imported/" + subject_id  # Saving to prepared_py
    input_path = "/data/p_02068/SRMR1_experiment/bids/" + subject_id + "/eeg/"  # Taking data from the bids folder
    # cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    montage_path = '/data/pt_02068/cfg/'
    montage_name = 'standard-10-5-cap385_added_mastoids.elp'
    os.makedirs(save_path, exist_ok=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    notch_freq = df.loc[df['var_name'] == 'notch_freq', 'var_value'].iloc[0]

    # Process ESG channels and then EEG channels separately
    # for esg_flag in [True, False]:  # True for esg, false for eeg
    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    stimulation = condition - 1

    # Set interpolation window
    # From Birgit's paper : 1.5 ms before to 4 ms after stimulus onset
    tstart_eng = -1.5/1000
    tmax_eng = 4/1000

    # Get file names that match pattern
    search = input_path + subject_id + '*' + cond_name + '*.set'
    cond_files = glob.glob(search)
    cond_files = sorted(cond_files)  # Arrange in order from lowest to highest value
    nblocks = len(cond_files)

    # Find out which channels are which, include ECG, exclude EOG
    eeg_chans, esg_chans, bipolar_chans = get_channels(subject_nr=subject, includesEcg=True, includesEog=True,
                                                       study_nr=srmr_nr)

    ####################################################################
    # Extract the raw data for each block, remove stimulus artefact, down-sample, concatenate, detect ecg,
    # and then save
    ####################################################################
    # Looping through each condition and each subject in ESG_Pipeline.py
    # Only dealing with one condition at a time, loop through however many blocks of said condition
    for iblock in np.arange(0, nblocks):
        # load data - need to read in files from EEGLAB format in bids folder
        fname = cond_files[iblock]
        raw = mne.io.read_raw_eeglab(fname, eog=(), preload=True, uint16_codec=None, verbose=None)

        # If you only want to look at bipolar channels
        raw.pick_channels(bipolar_chans)

        # Interpolate required channels
        # Only interpolate tibial, medial and alternating (conditions 2, 3, 4 ; stimulation 1, 2, 3)
        if stimulation != 0:

            # events contains timestamps with corresponding event_id
            # event_dict returns the event/trigger names with their corresponding event_id
            events, event_dict = mne.events_from_annotations(raw)

            # Fetch the event_id based on whether it was tibial/medial stimulation (trigger name)
            trigger_name = set(raw.annotations.description)

            # Acts in place to edit raw via linear interpolation to remove stimulus artefact
            # Need to loop as for alternating, there are 2 trigger names and event_ids at play
            for j in trigger_name:
                # Need to get indices of events linked to this trigger
                trigger_points = events[:, np.where(event_dict[j])]
                trigger_points = trigger_points.reshape(-1).reshape(-1)

                if repair_stim_art:
                    interpol_window = [tstart_eng, tmax_eng]
                    PCHIP_kwargs = dict(
                        debug_mode=False, interpol_window_sec=interpol_window,
                        trigger_indices=trigger_points, fs=sampling_rate_og
                    )
                    raw.apply_function(PCHIP_interpolation, picks=bipolar_chans, **PCHIP_kwargs,
                                       n_jobs=len(bipolar_chans))

        # Downsample the data
        # raw.resample(sampling_rate)  # resamples to desired

        # Append blocks of the same condition
        if iblock == 0:
            raw_concat = raw
        else:
            mne.concatenate_raws([raw_concat, raw])

    # Read .mat file with QRS events
    # Set filenames and append QRS annotations
    if repair_stim_art:
        fname_save = f'bipolar_repaired_{cond_name}.fif'
    else:
        fname_save = f'bipolar_{cond_name}.fif'

    ##############################################################################################
    # Reference and Remove Powerline Noise
    # High pass filter at 1Hz
    ##############################################################################################
    raw_concat.notch_filter(freqs=[notch_freq], n_jobs=len(raw_concat.ch_names), method='fir', phase='zero')

    raw_concat.filter(l_freq=1, h_freq=None, n_jobs=len(raw.ch_names), method='iir', iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    # Save data without stim artefact and downsampled to 1000
    raw_concat.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)
