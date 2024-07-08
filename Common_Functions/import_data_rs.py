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
        if esg_flag:
            input_path_trig = f"/data/pt_02718/tmp_data/freq_banded_esg/{subject_id}/"
        else:
            input_path_trig = f"/data/pt_02718/tmp_data/freq_banded_eeg/{subject_id}/"
        cond_names_trig = ['median', 'tibial']
        fnames_trig = [f"sigma_median.fif", 'sigma_tibial.fif']
        trigger_names_trig = ['Median - Stimulation', 'Tibial - Stimulation']
        os.makedirs(save_path, exist_ok=True)
        cond_name = cond_info.cond_name

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
        if esg_flag:
            input_path_trig = f"/data/pt_02718/tmp_data_2/freq_banded_esg/{subject_id}/"
        else:
            input_path_trig = f"/data/pt_02718/tmp_data_2/freq_banded_eeg/{subject_id}/"
        cond_names_trig = ['med_mixed', 'tib_mixed']
        fnames_trig = [f"sigma_med_mixed.fif", 'sigma_tib_mixed.fif']
        trigger_names_trig = ['medMixed', 'tibMixed']
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
        raise RuntimeError('Error: Check experiment number')

    cond_files = glob.glob(search)
    cond_files = sorted(cond_files)  # Arrange in order from lowest to highest value
    nblocks = len(cond_files)

    # Find out which channels are which, include ECG, exclude EOG
    eeg_chans, esg_chans, bipolar_chans = get_channels(subject_nr=subject, includesEcg=True, includesEog=True,
                                                       study_nr=srmr_nr)

    # Repeat for median and tibial conditions
    # First run will add tibial triggers to resting state file and save
    # Second run will add median triggers to resting state file and save
    for fname_trig, cond_name_trig, trigger_name_trig in zip(fnames_trig, cond_names_trig, trigger_names_trig):
        ####################################################################
        # Extract the raw data for each block, remove stimulus artefact, down-sample, concatenate, detect ecg,
        # and then save
        ####################################################################
        # Only dealing with one condition at a time, loop through however many blocks of said condition
        for iblock in np.arange(0, nblocks):
            # load in resting state data - need to read in files from EEGLAB format in bids folder
            fname = cond_files[iblock]
            raw = mne.io.read_raw_eeglab(fname, eog=(), preload=True, uint16_codec=None, verbose=None)

            # Drop channels of no interest
            if esg_flag:
                raw.pick_channels(esg_chans)
            else:
                raw.pick_channels(eeg_chans)

            # Downsample the data
            raw.resample(sampling_rate)  # resamples to desired

        raw_concat = raw  # Only one resting state file

        # Resting state files are WAY shorter than task-evoked files, going to replicate resting state *3
        # Repeat the resting state file 3 times before we add task-evoked triggers
        for i in np.arange(0, 3):
            mne.concatenate_raws([raw_concat, raw])

        # Load in task-evoked data
        raw_trig = mne.io.read_raw_fif(input_path_trig + fname_trig, preload=True)
        # event_dict returns the event/trigger names with their corresponding event_id
        events, event_dict = mne.events_from_annotations(raw_trig)

        # Since the task-evoked files already have qrs triggers added, we need to be sure we isolate only the stimulation triggers
        relevant_events = [list(event) for event in events if event[2]==event_dict[trigger_name_trig]]
        # Need to get indices of events linked to this trigger
        trigger_points = [event[0] for event in relevant_events]

        # Remove stimulus artefact and add identified triggers
        if esg_flag:
            interpol_window = [tstart_esg, tmax_esg]
            PCHIP_kwargs = dict(
                debug_mode=False, interpol_window_sec=interpol_window,
                trigger_indices=trigger_points, fs=sampling_rate
            )
            raw_concat.apply_function(PCHIP_interpolation, picks=esg_chans, **PCHIP_kwargs,
                               n_jobs=len(esg_chans))
            raw_concat.annotations.append([x / sampling_rate for x in trigger_points], 0.0, trigger_name_trig,
                                          ch_names=[esg_chans] * len(trigger_points))  # Add annotation
            fname_save = f'noStimart_sr{sampling_rate}_{cond_name}_{cond_name_trig}_withqrs.fif'


        elif not esg_flag:
            interpol_window = [tstart_eeg, tmax_eeg]
            PCHIP_kwargs = dict(
                debug_mode=False, interpol_window_sec=interpol_window,
                trigger_indices=trigger_points, fs=sampling_rate
            )
            raw_concat.apply_function(PCHIP_interpolation, picks=eeg_chans, **PCHIP_kwargs,
                               n_jobs=len(eeg_chans))
            raw_concat.annotations.append([x / sampling_rate for x in trigger_points], 0.0, trigger_name_trig,
                                          ch_names=[eeg_chans] * len(trigger_points))  # Add annotation
            fname_save = f'noStimart_sr{sampling_rate}_{cond_name}_{cond_name_trig}_withqrs_eeg.fif'

        else:
            raise RuntimeError('Flag has not been set - indicate if you are working with eeg or esg channels')

        ##############################################################################################
        # Add reference channel in case not in channel list and Remove Powerline Noise
        # High pass filter at 1Hz
        ##############################################################################################
        raw_concat.notch_filter(freqs=notch_freq, picks=raw_concat.ch_names, n_jobs=len(raw_concat.ch_names), method='fir', phase='zero')

        raw_concat.filter(l_freq=1, h_freq=None, n_jobs=len(raw_concat.ch_names), method='iir', iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

        # make sure recording reference is included
        mne.add_reference_channels(raw_concat, ref_channels=['TH6'], copy=False)  # Modifying in place, adds the channel but
        # doesn't do any actual rereferencing

        # Crop 5s after last relevant event
        # We do this just to avoid saving unecessarily large files
        t_event = trigger_points[-1] / sampling_rate
        raw_concat.crop(tmin= 0, tmax=t_event + 5)

        # Save data without stim artefact and downsampled to 5000
        raw_concat.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

        # Just deleting to be extra sure no crossover
        del raw, raw_concat
