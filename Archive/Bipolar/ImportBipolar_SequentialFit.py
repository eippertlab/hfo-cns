# Import necessary packages
import mne
from Common_Functions.get_conditioninfo import *
from Common_Functions.get_channels import *
from scipy.optimize import curve_fit
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# if __name__ == '__main__':
def import_dataepochs(subject, condition, srmr_nr, sampling_rate):
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
    notch_low = df.loc[df['var_name'] == 'notch_freq_low', 'var_value'].iloc[0]
    notch_high = df.loc[df['var_name'] == 'notch_freq_high', 'var_value'].iloc[0]
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    # Process ESG channels and then EEG channels separately
    # for esg_flag in [True, False]:  # True for esg, false for eeg
    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    stimulation = condition - 1

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
        # raw.resample(sampling_rate)  # resamples to desired

        # Append blocks of the same condition
        if iblock == 0:
            raw_concat = raw
        else:
            mne.concatenate_raws([raw_concat, raw])

    ##############################################################################################
    # Reference and Remove Powerline Noise
    # High pass filter at 1Hz
    ##############################################################################################
    # raw_concat.notch_filter(freqs=[notch_low, notch_high], n_jobs=len(raw_concat.ch_names), method='fir', phase='zero')
    #
    # raw_concat.filter(l_freq=1, h_freq=None, n_jobs=len(raw.ch_names), method='iir', iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    ##############################################################################################
    # Get epochs and do sequential fitting
    ##############################################################################################
    events, event_ids = mne.events_from_annotations(raw_concat)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw_concat, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                        baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)
    evoked = epochs.average()

    interpol_window = [3.5 / 1000, 20 / 1000]
    interpol_window1 = [3.5 / 1000, 7 / 1000]
    interpol_window2 = [13 / 1000, 20 / 1000]
    # Need to get list of all times between start and end point as indices
    index = epochs.time_as_index(interpol_window)
    index1 = epochs.time_as_index(interpol_window1)
    index2 = epochs.time_as_index(interpol_window2)
    times = epochs.times[index[0]:index[1]]
    times1 = epochs.times[index1[0]:index1[1]]
    times2 = epochs.times[index2[0]:index2[1]]

    # Want to fit to the average
    fits = {}
    for chan in evoked.ch_names:
        data = evoked.get_data(picks=chan).reshape(-1)

        popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(b * t) + c, times, data[index[0]:index[1]], p0=[1,0,1])
        a = popt[0]
        b = popt[1]
        c = popt[2]
        x_fitted = times
        y_fitted = a * np.exp(b * x_fitted) + c
        # print(data[index[0]:index[1]])
        # p = np.polyfit(times, np.log(data[index[0]:index[1]]), 1)
        # a = np.exp(p[1])
        # b = p[0]
        # x_fitted = times
        # y_fitted = a * np.exp(b * x_fitted)
        fits[f"fit_{chan}"] = y_fitted

        ax = plt.axes()
        # ax.scatter(data, label='Raw data')
        ax.plot(evoked.times, data)
        ax.scatter(times, data[index[0]:index[1]], label='Raw data')
        ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
        ax.set_title(f'{chan}')
        ax.set_ylabel('y-Values')
        ax.set_xlabel('x-Values')
        ax.legend()

        plt.show()

    exit()
    exponential_kwargs = dict(times=times, indices=index, all_time=epochs.times)
    evoked.apply_function(exponential_fit, picks=bipolar_chans, **exponential_kwargs,
                          )
    # n_jobs=len(bipolar_chans)


    # ##############################################################################################
    # # Save
    # ##############################################################################################
    # # Read .mat file with QRS events
    # # Set filenames and append QRS annotations
    # fname_save = f'bipolar_{cond_name}_exponentialepochs.fif'
    #
    # # Save data without stim artefact and downsampled to 1000
    # epochs.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)
