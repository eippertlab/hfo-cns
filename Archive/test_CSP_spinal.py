import mne
from scipy.io import loadmat
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
import os
import numpy as np
from meet.spatfilt import CSP
import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt

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
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    # Whole epoch is from -100ms to 300ms

    # Select the right files based on the data_string
    input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
    fname = f"{freq_band}_{cond_name}.fif"
    save_path = "/data/pt_02718/tmp_data/csp/" + subject_id + "/"
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
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                        baseline=tuple(iv_baseline), preload=True)

    # cca window size - Birgit created individual potential latencies for each subject
    fname_pot = 'potential_latency.mat'
    matdata = loadmat(potential_path + fname_pot)

    if cond_name == 'median':
        epochs = epochs.pick_channels(cervical_chans, ordered=True)
        esg_chans = cervical_chans
        sep_latency = matdata['med_potlatency']
        # window_times = [7/1000, 37/1000]
        window_times = [7 / 1000, 22 / 1000]
    elif cond_name == 'tibial':
        epochs = epochs.pick_channels(lumbar_chans, ordered=True)
        esg_chans = lumbar_chans
        sep_latency = matdata['tib_potlatency']
        # window_times = [7/1000, 47/1000]
        window_times = [15 / 1000, 30 / 1000]
    else:
        print('Invalid condition name attempted for use')
        exit()

    # Crop the epochs
    epo_csp = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)
    epo_contrast = epochs.copy().crop(tmin=80/1000, tmax=90/1000, include_tmax=False)

    # Need to then extract data - variables in rows, observations in columns, same as CCA
    epo_csp_data = epo_csp.get_data()
    epo_contrast_data = epo_contrast.get_data()
    st_matrix_csp = np.swapaxes(epo_csp_data, 1, 2).reshape(-1, epo_csp_data.shape[1]).T
    st_matrix_contrast = np.swapaxes(epo_contrast_data, 1, 2).reshape(-1, epo_contrast_data.shape[1]).T

    # Train filters
    filter, eigval = CSP(st_matrix_csp, st_matrix_contrast)
    # filtered_data = filter.T.dot(st_matrix_csp)

    # Apply filters to the whole epochs
    epoch_data = epochs.get_data()
    st_matrix = np.swapaxes(epoch_data, 1, 2).reshape(-1, epoch_data.shape[1]).T
    epoch_filtered_data = filter.T.dot(st_matrix)

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ filter

    # Reshape
    no_channels = len(epochs.ch_names)
    no_times_long = np.shape(epochs.get_data())[2]
    no_epochs = np.shape(epochs.get_data())[0]
    CSP_filtered = np.reshape(epoch_filtered_data, (no_channels, no_times_long, no_epochs), order='F')

    print(np.shape(CSP_filtered))

    ##########################################################################################
    # Save to epoch structure
    ##########################################################################################
    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CSP_comps = np.swapaxes(CSP_filtered, 0, 2)
    CSP_comps = np.swapaxes(CSP_comps, 1, 2)
    selected_channels = len(epochs.ch_names)  # Just keeping all for now to avoid rerunning

    data = CSP_comps[:, 0:selected_channels, :]
    events = epochs.events
    event_id = epochs.event_id
    tmin = iv_epoch[0]
    sfreq = 5000

    ch_names = []
    ch_types = []
    for i in np.arange(0, selected_channels):
        ch_names.append(epochs.ch_names[i])
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    csp_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    csp_epochs = csp_epochs.apply_baseline(baseline=tuple(iv_baseline))
    csp_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ##########################################################################################
    # Plot the time course - just testing median atm
    ##########################################################################################
    fig = plt.figure()
    data = csp_epochs.crop(tmin=0, tmax=65/1000).get_data(picks=['SC6'])
    to_plot = np.mean(data[:, 0, :], axis=0)
    plt.plot(csp_epochs.times, to_plot)
    plt.show()
    # plt.plot(csp_epochs.pick_channels(['SC6']).get_data())

