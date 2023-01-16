# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle


def keep_good_trials(subject, condition, srmr_nr, freq_band):
    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    # Select the right files based on the data_string
    input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
    fname = f"{freq_band}_{cond_name}.fif"
    save_path = "/data/pt_02718/tmp_data/good_trials_spinal/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23', 'TH6']

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_full = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)

    # potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
    # fname_pot = 'potential_latency.mat'
    # matdata = loadmat(potential_path + fname_pot)

    # Find indices of time period of interest
    if cond_name == 'median':
        # sep_latency = matdata['med_potlatency'][0][0]/1000
        # times = [sep_latency-2/1000, sep_latency+4/1000]
        times = [7/1000, 22/1000]
        epochs = epochs_full.copy().pick_channels(['SC6'])
        col = 'SC6'
    elif cond_name == 'tibial':
        # sep_latency = matdata['tib_potlatency'][0][0]/1000
        # times = [sep_latency-2/1000, sep_latency+4/1000]
        times = [15/1000, 30/1000]
        epochs = epochs_full.copy().pick_channels(['L1'])
        col = 'L1'

    # Get threshold for noise using all channels
    # cropped = epochs_full.copy().crop(tmin=-100/1000, tmax=-10/1000).pick_channels(esg_chans).get_data()
    # meanAllChan = np.mean(cropped, axis=0)
    # amplitudeThreshold = np.max(abs(meanAllChan))*2*10**6

    # Trying noise threshold as std with only channel of interest with just channel of interest
    cropped = epochs.copy().crop(tmin=-100/1000, tmax=-10/1000).get_data()
    mean = cropped.mean()
    std = cropped.std()
    amplitudeThreshold = np.max([abs(mean-3*std), abs(mean+3*std)])*10**6

    indices = epochs.time_as_index(times)

    df = epochs.to_data_frame()
    keep = []
    count = 0
    for i in set(df.epoch):
        sub_df = df[df.epoch == i]
        vals = sub_df[sub_df["time"].between(times[0], times[1])][col]
        # plt.figure()
        # plt.plot(sub_df['time'], sub_df[col])
        keep.append(np.any(abs(vals) > amplitudeThreshold))
        # count += 1
        # if count == 4:
        #     plt.show()
        #     exit()

    # Now we want to save these indices
    afile = open(save_path + f'good_{freq_band}_{cond_name}_strict.pkl', 'wb')
    pickle.dump(keep, afile)
    afile.close()

    # print(keep)
    # print(amplitudeThreshold)
    # check = [x for x in keep if x == True]
    # print(len(check))
    # exit()
