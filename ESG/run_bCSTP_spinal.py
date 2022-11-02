# To implement bCSTP as in the Neuorphysics/meet package

import mne
import random
from meet.spatfilt import bCSTP
from scipy.io import loadmat
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
import os
import numpy as np
import h5py
import pickle

if __name__ == '__main__':
    # For testing
    condition = 2
    srmr_nr = 1
    subject = 1
    freq_band = 'sigma'
    s_freq = 5000

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
    iv_epoch = [df.loc[df['var_name'] == 'epo_bcstp_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_bcstp_end', 'var_value'].iloc[0]]

    # Select the right files based on the data_string
    input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
    fname = f"{freq_band}_{cond_name}.fif"
    save_path = "/data/pt_02718/tmp_data/bCSTP/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23', 'TH6']

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # create c+ epochs from -40ms to 90ms
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_pos = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline), preload=True)

    # create c- epochs with random onsets and 130ms length
    # Create random event array
    max_time = int(np.floor(raw.n_times - s_freq*0.15))
    no_epochs = np.shape(epochs_pos.get_data())[0]
    event_times = random.sample(range(1, max_time), no_epochs+1)  # Need to be all unique to avoid selecting the same trials
    event_times.sort()  # Want in chronological order
    event_array = np.zeros((no_epochs, 3), dtype=np.int32)
    for no in np.arange(0, no_epochs):
        event_array[no][0] = event_times[no]
        event_array[no][1] = int(0)
        event_array[no][2] = int(event_id_dict[trigger_name])

    epochs_neg = mne.Epochs(raw, event_array, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                            baseline=tuple(iv_baseline), preload=True, reject=None, flat=None,
                            reject_by_annotation=False)

    # Select only the ESG channels
    epochs_pos = epochs_pos.pick_channels(ch_names=esg_chans)
    epochs_neg = epochs_neg.pick_channels(ch_names=esg_chans)

    # Change order of data
    data_pos = np.transpose(epochs_pos.get_data(), (1, 2, 0))
    data_neg = np.transpose(epochs_neg.get_data(), (1, 2, 0))

    # Run bCSTP, leaving other vars at default
    W, V, s_eigvals, t_eigvals = bCSTP(data_pos, data_neg, num_iter=10)

    ## SAVE so I can play in sandbox without waiting and figure it out ##
    count = 0
    name = ['W', 'V', 's_eigvals', 't_eigvals']
    for obj in [W, V, s_eigvals, t_eigvals]:
        afile = open(f'/data/pt_02718/{name[count]}.pkl', 'wb')
        pickle.dump(obj, afile)
        afile.close()
        count += 1
    exit()

    SF_final = W[-1]
    TF_final = V[-1]
    s_eigvals_final = s_eigvals[-1]
    t_eigvals_final = t_eigvals[-1]

    # Invert final filters to get final patterns
    # Each row of these gives a different temporal and spatial pattern
    CSP = np.linalg.inv(SF_final)
    CTP = np.linalg.inv(TF_final)

    # V[-1] is most updated temporal filters, invert for temporal pattern
    # Each row of V⁻¹ (CTP) is a common temporal pattern
    # Rows of W⁻¹ (CSP) are common spatial patterns

    # Plot the first 4 CTP and CSP

    # Apply strongest spatial filter to single trials and then plot

    # Get average across single trials and plot

    # cca window size - Birgit created individual potential latencies for each subject
    fname_pot = 'potential_latency.mat'
    matdata = loadmat(potential_path + fname_pot)
