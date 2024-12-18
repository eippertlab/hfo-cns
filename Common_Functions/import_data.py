##########################################################################################
#                               This Script
# 1) imports the blocks based on the condition name in EEGLAB form from the BIDS directory
# 2) removes the stimulus artifact iv: -1.5 to 6 ms, for ESG use -7 to 7s - linear interpolation
# 3) downsample the signal to 5000 Hz
# 4) Append mne raws of the same condition
# 5) Add qrs events as annotations
# 6) saves the new raw structure
# Emma Bailey, October 2022
# .matfile with R-peak locations is at 1000Hz - will still give rough idea (annotations not used)
##########################################################################################

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


def import_data(subject, condition, srmr_nr, sampling_rate, esg_flag):
    subject_id = f'sub-{str(subject).zfill(3)}'
    cond_info = get_conditioninfo(condition, srmr_nr)

    if srmr_nr == 1:
        save_path = "../tmp_data/imported/" + subject_id
        input_path = "/data/p_02068/SRMR1_experiment/bids/" + subject_id + "/eeg/"  # Taking data from the bids folder
        montage_path = '/data/pt_02068/cfg/'
        montage_name = 'standard-10-5-cap385_added_mastoids.elp'
        os.makedirs(save_path, exist_ok=True)
        cond_name = cond_info.cond_name
        stimulation = condition - 1

    elif srmr_nr == 2:
        save_path = "../tmp_data_2/imported/" + subject_id
        input_path = "/data/p_02151/SRMR2_experiment/bids/" + subject_id + "/eeg/"  # Taking data from the bids folder
        os.makedirs(save_path, exist_ok=True)
        if condition == 1:
            cond_name = cond_info.cond_name
            cond_name2 = cond_info.cond_name  # rest
        else:
            cond_name = cond_info.cond_name   # med_digits/mixed and tib_digits/mixed
            cond_name2 = cond_info.cond_name2  # mediansensory/mixed or tibialsensory/mixed
        stimulation = condition - 1
    else:
        print('Error: Experiment 1 or experiment 2 must be specified')
        exit()

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    notch_freq = df.loc[df['var_name'] == 'notch_freq', 'var_value'].iloc[0]

    sampling_rate_og = 10000

    # Set interpolation window (different for eeg and esg data, both in seconds)
    tstart_esg = -0.007
    tmax_esg = 0.007

    tstart_eeg = -0.0015
    tmax_eeg = 0.006

    # Get file names that match pattern
    if srmr_nr == 1:
        search = input_path + subject_id + '*' + cond_name + '*.set'
    elif srmr_nr == 2:
        search = input_path + subject_id + '*' + cond_name2 + '*.set'
    else:
        print('Error: Check experiment number')
        exit()
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

        # If you only want to look at esg channels, drop the rest
        if esg_flag:
            raw.pick_channels(esg_chans)
        else:
            raw.pick_channels(eeg_chans)

        # Interpolate required channels
        # Only interpolate tibial, medial and alternating (conditions 2, 3, 4 ; stimulation 1, 2, 3)
        if stimulation != 0:

            # events contains timestamps with corresponding event_id
            # event_dict returns the event/trigger names with their corresponding event_id
            events, event_dict = mne.events_from_annotations(raw)

            # Fetch the event_id based on whether it was tibial/medial stimulation (trigger name)
            trigger_name = set(raw.annotations.description)

            # Acts in place to edit raw via linear interpolation to remove stimulus artefact
            # Loop is not really necessary - won't work for alternating stimulation
            for j in trigger_name:
                # Need to get indices of events linked to this trigger
                trigger_points = events[:, np.where(event_dict[j])]
                trigger_points = trigger_points.reshape(-1).reshape(-1)

                if esg_flag:
                    interpol_window = [tstart_esg, tmax_esg]
                    PCHIP_kwargs = dict(
                        debug_mode=False, interpol_window_sec=interpol_window,
                        trigger_indices=trigger_points, fs=sampling_rate_og
                    )
                    raw.apply_function(PCHIP_interpolation, picks=esg_chans, **PCHIP_kwargs,
                                       n_jobs=len(esg_chans))
                    # mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[j], tmin=tstart_esg,
                    #                                     tmax=tmax_esg, mode='linear', stim_channel=None)

                elif not esg_flag:
                    interpol_window = [tstart_eeg, tmax_eeg]
                    PCHIP_kwargs = dict(
                        debug_mode=False, interpol_window_sec=interpol_window,
                        trigger_indices=trigger_points, fs=sampling_rate_og
                    )
                    raw.apply_function(PCHIP_interpolation, picks=eeg_chans, **PCHIP_kwargs,
                                       n_jobs=len(eeg_chans))
                    # mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[j], tmin=tstart_eeg,
                    #                                     tmax=tmax_eeg, mode='linear', stim_channel=None)

                else:
                    print('Flag has not been set - indicate if you are working with eeg or esg channels')

        # Downsample the data
        raw.resample(sampling_rate)  # resamples to desired

        # Append blocks of the same condition
        if iblock == 0:
            raw_concat = raw
        else:
            mne.concatenate_raws([raw_concat, raw])

    if stimulation != 0:
        # Read .mat file with QRS events - don't use these anymore since we use SSP but adding for completeness
        if srmr_nr == 1:
            input_path_m = "/data/pt_02718/Rpeaks/" + subject_id + "/"
            fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
        elif srmr_nr == 2:
            input_path_m = "/data/pt_02718/Rpeaks_2/" + subject_id + "/"
            fname_m = f"raw_{sampling_rate}_spinal_{cond_name}"
        else:
            print('Error: Experiment 1 or 2 must be specified')
            exit()

        matdata = loadmat(input_path_m + fname_m + '.mat')
        QRSevents_m = matdata['QRSevents'][0]

        # Add qrs events as annotations
        qrs_event = [x / sampling_rate for x in QRSevents_m]  # Divide by sampling rate to make times
        duration = np.repeat(0.0, len(QRSevents_m))
        description = ['qrs'] * len(QRSevents_m)

    # Set filenames and append QRS annotations
    if esg_flag:
        if stimulation != 0:
            raw_concat.annotations.append(qrs_event, duration, description, ch_names=[esg_chans] * len(QRSevents_m))
        fname_save = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs.fif'
    else:
        if stimulation != 0:
            raw_concat.annotations.append(qrs_event, duration, description, ch_names=[eeg_chans] * len(QRSevents_m))
        fname_save = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif'

    ##############################################################################################
    # Add reference channel in case not in channel list and Remove Powerline Noise
    # High pass filter at 1Hz
    ##############################################################################################
    raw_concat.notch_filter(freqs=notch_freq, picks=raw_concat.ch_names, n_jobs=len(raw_concat.ch_names), method='fir', phase='zero')

    raw_concat.filter(l_freq=1, h_freq=None, n_jobs=len(raw.ch_names), method='iir', iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    # make sure recording reference is included
    mne.add_reference_channels(raw_concat, ref_channels=['TH6'], copy=False)  # Modifying in place, adds the channel but
    # doesn't do any actual rereferencing

    # Save data without stim artefact and downsampled to 1000
    raw_concat.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

