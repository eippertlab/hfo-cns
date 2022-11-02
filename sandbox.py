import mne
import random
from meet.spatfilt import bCSTP
from scipy.io import loadmat
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
import os
import numpy as np
import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot

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
    input_path = "/data/pt_02718/tmp_data/freq_banded/" + subject_id + "/"
    fname = f"{freq_band}_{cond_name}.fif"
    save_path = "/data/p_02718/tmp_data/bCSTP/" + subject_id + "/"
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

    # Select only the ESG channels
    epochs_pos = epochs_pos.pick_channels(ch_names=esg_chans)

    # Change order of data
    data_pos = np.transpose(epochs_pos.get_data(), (1, 2, 0))
    # This is what we submitted into the bCSTP algorithm

    # Load the outputs from the algorithm
    names = ['W', 'V', 's_eigvals', 't_eigvals']
    for name in names:
        afile = open(f'/data/pt_02718/{name}.pkl', 'rb')
        obj = pickle.load(afile)
        afile.close()
        if name == 'W':
            W = obj
        elif name == 'V':
            V = obj
        elif name == 's_eigvals':
            s_eigvals = obj
        elif name == 't_eigvals':
            t_eigvals = obj

    # Each one is a list of arrays corresponding to an iteration of the algorithm
    # Use [-1] to access the final ones
    SF_final = W[-1]
    TF_final = V[-1]
    s_eigvals_final = s_eigvals[-1]
    t_eigvals_final = t_eigvals[-1]

    # print(np.shape(SF_final))  # (40, 39) - (no_channels,  )
    # print(np.shape(TF_final))  # (701, 694) - (no_timepoints,  )
    # print(np.shape(data_pos))  # (40, 701, 2000) - (no_channels, no_timepoints, no_epochs)

    # Filter matrices are not square
    CSP = np.linalg.pinv(SF_final)
    CTP = np.linalg.pinv(TF_final)

    # print(np.shape(CSP))  # (39, 40) - (no_filters , no_channel)
    # print(np.shape(CTP))  # (694, 701)  - (no_filters  , no_timepoints)
    # CSP = np.linalg.inv(SF_final)
    # CTP = np.linalg.inv(TF_final)

    # This loop gets the optimum spatio-temporal features for each trial
    # Need to then prune the temporal patterns and project back to original space
    spatio_temporal = []
    for trial in np.arange(0, 2000):
        trial_selected = data_pos[:, :, trial]  # will be (no_channels x no_timepoints)
        trial_cleaned = (SF_final.T).dot(trial_selected).dot(TF_final)
        spatio_temporal.append(trial_cleaned)
    print(np.shape(spatio_temporal))
    print(np.shape(spatio_temporal[0]))
    exit()
    test_data = data_pos.reshape(-1, data_pos.shape[0]).T
    test_shape = (SF_final.T).dot(data_pos).dot(TF_final)
    # test_shape = (SF_final.T.dot(test_data)).dot(TF_final)
    print(np.shape(test_shape))
    exit()
    all_filters = len(s_eigvals[-1])
    # data_pos is (no_channels, no_times, no_epochs), want (no_channels, no_times x no_epochs)
    # test_filters is all filters (no_channels, no_filters)
    test_data = data_pos.reshape(-1, data_pos.shape[0]).T
    test_filters = SF_final[:, 0:all_filters]
    cleaned_single_trials = test_data.T @ test_filters
    no_times_long = np.shape(epochs_pos.get_data())[2]
    no_epochs = np.shape(epochs_pos.get_data())[0]
    bCSTP_single = np.reshape(cleaned_single_trials, (all_filters, no_times_long, no_epochs), order='F')  # Perform reshape
    # This is finally (no_filters, no_times, no_epochs)

    # Get into form to create epochs in MNE
    bCSTP_single = np.swapaxes(bCSTP_single, 0, 2)
    bCSTP_single = np.swapaxes(bCSTP_single, 1, 2)

    data = bCSTP_single[:, :, :]
    events = epochs_pos.events
    event_id = epochs_pos.event_id
    tmin = iv_epoch[0]
    sfreq = 5000

    ch_names = []
    ch_types = []
    for i in np.arange(0, all_filters):
        ch_names.append(f'Cor{i + 1}')
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    bCSTP_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    bCSTP_epochs = bCSTP_epochs.apply_baseline(baseline=tuple(iv_baseline))
    save_path = '/data/pt_02718/'
    fname = 'test_epochs.fif'
    bCSTP_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ##############################################################################
    # PLOTTING
    ##############################################################################
    ######################## Plot image for cca_epochs ############################
    # cca_epochs and cca_epochs_d both already baseline corrected before this point
    figure_path_st = f'/data/p_02718/Images/CCA/ComponentSinglePlots/{subject_id}/'
    os.makedirs(figure_path_st, exist_ok=True)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    axes = [ax1, ax2, ax3, ax4]
    cropped = bCSTP_epochs.copy().crop(tmin=-0.025, tmax=0.065)
    cmap = mpl.colors.ListedColormap(["blue", "green", "red"])
    vmin = -10000
    vmax = 10000

    for icomp in np.arange(0, 4):
        cropped.plot_image(picks=f'Cor{icomp + 1}', combine=None, cmap='jet', evoked=False, show=False,
                           axes=axes[icomp], title=f'Component {icomp + 1}', colorbar=False, group_by=None,
                           vmin=vmin, vmax=vmax, units=dict(eeg='V'), scalings=dict(eeg=1))

    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm, spacing='proportional')
    mpl.colorbar.ColorbarBase(ax5, cmap='jet', norm=norm)
    # has to be as a list - starts with x, y coordinates for start and then width and height in % of figure width
    # plt.savefig(figure_path_st + f'{freq_band}_{cond_name}.png')
    # plt.close(fig)
    plt.show()
    exit()

    ############ Time Course of average of first 5 ###############
    fig = plt.figure()
    data = bCSTP_epochs.get_data(picks=['Cor1', 'Cor2', 'Cor3', 'Cor4', 'Cor5'])
    to_plot = np.mean(data[:, 0:5, :], axis=tuple([0, 1]))
    plt.plot(bCSTP_epochs.times, to_plot)
    plt.xlim([-0.025, 0.065])
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [A.U.]')

    # Plot average of first 5 temporal patterns
    fig = plt.figure()
    plt.plot(np.mean(CTP[0:5, :], axis=0))
    plt.show()
    exit()

    # Plot first 4 spatial patterns
    ####### Spinal Isopotential Plots for the first 4 components ########
    # fig, axes = plt.figure()
    fig, axes = plt.subplots(2, 2)
    axes_unflat = axes
    axes = axes.flatten()
    for icomp in np.arange(0, 4):  # Plot for each of four components
        if freq_band == 'sigma':
            colorbar_axes = [-0.3, 0.3]
        else:
            colorbar_axes = [-0.1, 0.1]
        chan_labels = epochs_pos.ch_names
        colorbar = True
        time = 0.0
        mrmr_esg_isopotentialplot([subject], CSP[icomp, :], colorbar_axes, chan_labels,
                                  colorbar, time, axes[icomp])
        axes[icomp].set_title(f'Component {icomp + 1}')

    plt.show()
