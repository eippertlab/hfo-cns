##########################################################################################
# Emma Bailey, July 6th 2023
# Script to import data, downsample, remove stim artefact, apply SSP and then run bad channel check
##########################################################################################

# Import necessary packages
import mne
from Common_Functions.get_conditioninfo import *
from Common_Functions.get_channels import *
import os
import glob
import numpy as np
import pandas as pd
from Common_Functions.pchip_interpolation import PCHIP_interpolation
import matplotlib.pyplot as plt
from datetime import datetime


if __name__ == '__main__':
    startTime = datetime.now()
    # Set subject, condition number (2-median, 3-tibial), and study number (I wrote the code to handle 2 exps)
    subject = 1
    condition = 2
    srmr_nr = 1

    subject_id = f'sub-{str(subject).zfill(3)}'
    cond_info = get_conditioninfo(condition, srmr_nr)  # Just calls another script to get actual name of condition
    save_path = "/data/p_02718/test_lisa/imported/" + subject_id
    input_path = "/data/p_02068/SRMR1_experiment/bids/" + subject_id + "/eeg/"  # Taking data from the bids folder
    os.makedirs(save_path, exist_ok=True)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    stimulation = condition - 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment (frequencies for filtering, timing
    # for epoching etc
    df = pd.read_excel(cfg_path)
    notch_low = df.loc[df['var_name'] == 'notch_freq_low', 'var_value'].iloc[0]
    notch_high = df.loc[df['var_name'] == 'notch_freq_high', 'var_value'].iloc[0]
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    sampling_rate_og = 10000  # Original sampling frequency (in Hz)
    sampling_rate = 1000  # Frequency to downsample to (in Hz)

    # Set interpolation window (in seconds)
    tstart_esg = -0.007
    tmax_esg = 0.007

    # Get file names that match pattern (I take all the blocks together for this analysis so I locate the files and then
    # arrange them in chronological order for concatenation)
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

        # If you only want to look at esg channels, drop the rest
        raw.pick_channels(esg_chans)

        # Interpolate required channels
        # Only interpolate tibial, medial and alternating (conditions 2, 3, 4 ; stimulation 1, 2, 3)
        if stimulation != 0:

            # events contains timestamps with corresponding event_id
            # event_dict returns the event/trigger names with their corresponding event_id
            events, event_dict = mne.events_from_annotations(raw)

            # Fetch the event_id based on whether it was tibial/medial stimulation (trigger name)
            trigger_raw = set(raw.annotations.description)

            # Acts in place to edit raw via linear interpolation to remove stimulus artefact
            # Need to loop as for alternating, there are 2 trigger names and event_ids at play
            for j in trigger_raw:
                # Need to get indices of events linked to this trigger
                trigger_points = events[:, np.where(event_dict[j])]
                trigger_points = trigger_points.reshape(-1).reshape(-1)

                interpol_window = [tstart_esg, tmax_esg]
                # PCHIP works better than inbuilt MNE functions so I apply this
                PCHIP_kwargs = dict(
                    debug_mode=False, interpol_window_sec=interpol_window,
                    trigger_indices=trigger_points, fs=sampling_rate_og
                )
                raw.apply_function(PCHIP_interpolation, picks=esg_chans, **PCHIP_kwargs,
                                   n_jobs=len(esg_chans))
                # mne.preprocessing.fix_stim_artifact(raw, events=events, event_id=event_dict[j], tmin=tstart_esg,
                #                                     tmax=tmax_esg, mode='linear', stim_channel=None)

        # Downsample the data
        raw.resample(sampling_rate)  # resamples to desired

        # Append blocks of the same condition
        if iblock == 0:
            raw_concat = raw
        else:
            mne.concatenate_raws([raw_concat, raw])

    ##############################################################################################
    # Re-reference and Remove Powerline Noise
    # High pass filter at 1Hz
    ##############################################################################################
    # make sure recording reference is included
    mne.add_reference_channels(raw_concat, ref_channels=['TH6'], copy=False)  # Modifying in place

    raw_concat.notch_filter(freqs=[notch_low, notch_high], n_jobs=len(raw_concat.ch_names), method='fir', phase='zero')

    raw_concat.filter(l_freq=1, h_freq=None, n_jobs=len(raw.ch_names), method='iir',
                      iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    # Save data without stim artefact and downsampled to 1000
    # raw_concat.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)

    ##########################################################################################
    # SSP
    ##########################################################################################
    n_p = 6  # I remove 6 projections but you can change this as you want
    projs, events = mne.preprocessing.compute_proj_ecg(raw_concat, n_eeg=n_p, reject=None,
                                                       n_jobs=len(raw.ch_names), ch_name='ECG')

    # Apply projections (clean data)
    clean_raw = raw_concat.copy().add_proj(projs)
    clean_raw = clean_raw.apply_proj()

    ##############################################################################################
    # Bad channel check
    ##############################################################################################
    ##########################################################################################
    # Generates psd - can click on plot to find bad channel name
    ##########################################################################################
    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Power spectral density, data sub-{subject}")
    fig.tight_layout(pad=3.0)
    if 'TH6' in clean_raw.ch_names:  # Can't use zero value in spectrum for channel
        clean_raw.copy().drop_channels('TH6').compute_psd(fmax=500).plot(axes=axes, show=False)
    else:
        clean_raw.compute_psd(fmax=500).plot(axes=axes, show=False)
    axes.set_ylim([-80, 50])
    # plt.savefig(figure_path + f'psd_{cond_name}.png')

    ###########################################################################################
    # Squared log means of each channel
    ###########################################################################################
    events, event_ids = mne.events_from_annotations(clean_raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(clean_raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline), preload=True)
    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Squared log means per epoch sub-{subject}")
    table = epochs.to_data_frame()
    table = table.drop(columns=["time", "ECG", "TH6"])
    table = pd.concat([table.iloc[:, :2], np.square(table.iloc[:, 2:])], axis=1)
    table = pd.concat([table.iloc[:, :2], np.log(table.iloc[:, 2:])], axis=1)
    means = table.groupby(['epoch']).mean().T  # average
    ax_i = axes.matshow(means, aspect='auto')  # plots mean values by colorscale
    plt.colorbar(ax_i, ax=axes)
    axes.set_yticks(np.arange(0, len(list(means.index))), list(means.index))  # Don't hardcode 41
    axes.tick_params(labelbottom=True)
    # plt.savefig(figure_path + f'meanlog_{cond_name}.png')
    print(datetime.now() - startTime)
    plt.show()

    # I examine these in realtime and enter bad channels to be saved to a text file but you probably won't do this
    # bad_chans = list(map(str, input("Enter bad channels (separated by a space, press enter if none): ").split()))
    # filename = figure_path + f'bad_channels_{cond_name}.txt'
    # with open(filename, mode="w") as outfile:
    #     for s in bad_chans:
    #         outfile.write("%s\n" % s)
    #
    # if bad_chans:  # If there's something in here append it to the bads
    #     clean_raw.info['bads'].extend(bad_chans)
    #
    #     # Save them with the bad channels now marked
    #     clean_raw.save(f"{input_path}{fname}", fmt='double', overwrite=True)

