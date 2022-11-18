import mne
import random
from meet.spatfilt import bCSTP
from scipy.io import loadmat
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import os
import numpy as np
from meet._cSPoC import pattern_from_filter
import scipy
import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot

if __name__ == '__main__':
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
    input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
    fname = f"{freq_band}_{cond_name}.fif"
    save_path = "/data/pt_02718/tmp_data/bCSTP_eeg/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Set montage
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

    # create c+ epochs from -40ms to 90ms
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_pos = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                            baseline=tuple(iv_baseline), preload=True)
    epochs_pos.set_eeg_reference(ref_channels=['C4'])
    epochs_pos = epochs_pos.pick_channels(ch_names=eeg_chans)
    data_pos = np.transpose(epochs_pos.get_data(), (1, 2, 0))

    # Read in W, V
    with open(f'/data/pt_02718/W_{subject_id}.pkl', 'rb') as f:
        W = pickle.load(f)
        # Shape (channels, channel_rank)

    with open(f'/data/pt_02718/V_{subject_id}.pkl', 'rb') as f:
        V = pickle.load(f)
        # Shape (epoch_length, epoch_rank)

    with open(f'/data/pt_02718/s_eigvals_{subject_id}.pkl', 'rb') as f:
        s_eigvals = pickle.load(f)

    with open(f'/data/pt_02718/t_eigvals_{subject_id}.pkl', 'rb') as f:
        t_eigvals = pickle.load(f)

    # print(s_eigvals[-1])
    # print(t_eigvals[-1])
    # exit()
    keeps = 1
    keept = 5
    spatial_filters = W[-1][:, 0:keeps]  # Keep top spatial filters
    temporal_filters = V[-1][:, 0:keept]  # Keep top temporal filters
    # print(np.shape(data_pos))
    spatially_filtered_trials = np.tensordot(spatial_filters, data_pos, axes=(0, 0))
    # print(np.shape(spatially_filtered_trials))
    temporally_filtered_trials = np.tensordot(temporal_filters, data_pos, axes=(0, 1))
    # print(np.shape(temporally_filtered_trials))
    # exit()
    spatio_temp_filtered_trials = np.tensordot(temporal_filters, spatially_filtered_trials, axes=(0, 1))
    # print(np.shape(spatio_temp_filtered_trials))
    # Tells us how much activity in each trial corresponds to spatiotemporal patterns

    # Recover patterns
    temporal_patterns = pattern_from_filter(temporal_filters, np.transpose(data_pos, (1, 0, 2)))
    spatial_patterns = pattern_from_filter(spatial_filters, data_pos)
    # print(np.shape(temporal_patterns))
    # print(np.shape(spatial_patterns))

    # Reconstructed data
    reconstructed_trials_temporal = np.tensordot(temporal_patterns, spatio_temp_filtered_trials, axes=(0, 0))
    # print(np.shape(reconstructed_trials_temporal))
    reconstructed_trials_spatial = np.tensordot(spatial_patterns, spatio_temp_filtered_trials, axes=(0, 1))
    # print(np.shape(reconstructed_trials_spatial))
    reconstructed_trials = np.tensordot(spatial_patterns, reconstructed_trials_temporal, axes=(0, 1))
    # print(np.shape(reconstructed_trials))
    # exit()

    #######################  Epoch data class to store the information ####################
    events = epochs_pos.events
    event_id = epochs_pos.event_id
    tmin = iv_epoch[0]
    sfreq = 5000

    ch_names = epochs_pos.ch_names
    ch_types = ['eeg' for i in np.arange(0, len(ch_names))]

    # Initialize an info structure
    info_full_recon = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    bcstp_epochs = mne.EpochsArray(np.transpose(reconstructed_trials, axes=(2, 0, 1)), info_full_recon, events, tmin,
                                   event_id)
    bcstp_epochs = bcstp_epochs.apply_baseline(baseline=tuple(iv_baseline))
    bcstp_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ################################ Plotting Graphs #######################################
    figure_path_spatial = f'/data/p_02718/Images/bCSTP_eeg/IsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)

    plot_graphs = True
    if plot_graphs:
        ####### Topography for the best spatial pattern ########
        # fig, axes = plt.figure()
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig, axes = plt.subplots(1, 1)
        # axes_unflat = axes
        # axes = axes.flatten()
        # for icomp in np.arange(0, 1):  # Plot for each of four components
        mne.viz.plot_topomap(data=spatial_patterns[0, :], pos=epochs_pos.info, ch_type='eeg', sensors=True,
                             names=epochs_pos.ch_names,
                             contours=6, outlines='head', sphere=None, image_interp='cubic',
                             extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                             cnorm=None, axes=axes, show=False)
        axes.set_title(f'Component {1}')
        divider = make_axes_locatable(axes)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(axes.images[-1], cax=cax, shrink=0.6, orientation='vertical')

        ############### Time Course of best Temporal Patterns ###########################
        fig, axes = plt.subplots(2, 2)
        for count, ax in enumerate(axes.flatten()):
            data = temporal_patterns[count, :]
            times = epochs_pos.times
            ax.plot(times, data)
            ax.set_title(f'Temporal Pattern {count}')


        ############### Time Course of Reconstructed in Temporal Domain #################
        # data = np.mean(reconstructed_trials, axis=(0, 2))
        # plt.figure()
        # plt.plot(times, data)
        # plt.xlim([0.01, 0.050])
        # plt.xlabel('Time (ms)')
        # plt.title('Temporally Reconstructed Data')
        # plt.savefig(figure_path_spatial + f'{freq_band}_{cond_name}.png')
        # plt.close(fig)

        ############ Evoked re-construction ####################################
        bcstp_epochs.set_montage(montage, on_missing="ignore")
        bcstp_evoked = bcstp_epochs.average()
        bcstp_evoked.plot()
        bcstp_evoked.plot_topomap(times=[0.015, 0.02, 0.025], average=0.005)
        bcstp_evoked.plot_joint()

        ############ Single Trial Plot of Reconstructed data ###################
        # for channel in bcstp_epochs.ch_names:
        #     bcstp_epochs.crop(tmin=0.01, tmax=0.05).plot_image(picks=[channel], scalings=dict(eeg=1), title=channel)


        ######################## Plot image for bCSTP_epochs at each channel ############################
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        # figure_path_st = f'/data/p_02718/Images/bCSTP_eeg/SingleTrialPlots/{subject_id}/'
        # os.makedirs(figure_path_st, exist_ok=True)

        # fig = plt.figure()
        # channels = ['FCz']
        # for channel in channels:
        #     bcstp_epochs.crop(tmin=0.01, tmax=0.05).plot_image(picks=[channel], scalings=dict(eeg=1), title=channel)
        # layout = mne.channels.find_layout(epochs_pos.info, ch_type='eeg')
        # bcstp_epochs.plot_topo_image(layout=layout, fig_facecolor='w', font_color='k', sigma=1)

    plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     