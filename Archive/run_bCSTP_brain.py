# To implement bCSTP as in the Neuorphysics/meet package

import mne
import random
from meet.spatfilt import bCSTP
from meet._cSPoC import pattern_from_filter
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle


def run_bCSTP(subject, condition, srmr_nr, freq_band, sfreq):
# if __name__ == '__main__':
#     # For testing
#     condition = 2
#     srmr_nr = 1
#     subject = 1
#     freq_band = 'sigma'
#     s_freq = 5000

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

    # create c- epochs with random onsets and 130ms length
    # Create random event array
    max_time = int(np.floor(raw.n_times - sfreq*0.15))
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

    # Select only the EEG channels
    epochs_pos = epochs_pos.pick_channels(ch_names=eeg_chans)
    epochs_neg = epochs_neg.pick_channels(ch_names=eeg_chans)

    # Change order of data
    data_pos = np.transpose(epochs_pos.get_data(), (1, 2, 0))
    data_neg = np.transpose(epochs_neg.get_data(), (1, 2, 0))

    # Run bCSTP, leaving other vars at default
    W, V, s_eigvals, t_eigvals = bCSTP(data_pos, data_neg, num_iter=10)

    ## SAVE so I can play in sandbox without waiting and figure it out ##
    count = 0
    name = ['W', 'V', 's_eigvals', 't_eigvals']
    for obj in [W, V, s_eigvals, t_eigvals]:
        afile = open(save_path + f'{name[count]}_{freq_band}_{cond_name}.pkl', 'wb')
        pickle.dump(obj, afile)
        afile.close()
        count += 1

    # Define number of each type of filter to keep
    keeps = 1
    if freq_band == 'sigma':
        keept = 5
    elif freq_band == 'kappa':
        keept = 6

    spatial_filters = W[-1][:, 0:keeps]  # Keep top spatial filters
    temporal_filters = V[-1][:, 0:keept]  # Keep top temporal filters

    # Apply filters
    spatially_filtered_trials = np.tensordot(spatial_filters, data_pos, axes=(0, 0))
    temporally_filtered_trials = np.tensordot(temporal_filters, data_pos, axes=(0, 1))
    spatio_temp_filtered_trials = np.tensordot(temporal_filters, spatially_filtered_trials, axes=(0, 1))
    # Tells us how much activity in each trial corresponds to spatiotemporal patterns

    # Recover patterns
    temporal_patterns = pattern_from_filter(temporal_filters, np.transpose(data_pos, (1, 0, 2)))
    spatial_patterns = pattern_from_filter(spatial_filters, data_pos)

    # Reconstructed data
    reconstructed_trials_temporal = np.tensordot(temporal_patterns, spatio_temp_filtered_trials, axes=(0, 0))
    reconstructed_trials_spatial = np.tensordot(spatial_patterns, spatio_temp_filtered_trials, axes=(0, 1))
    reconstructed_trials = np.tensordot(spatial_patterns, reconstructed_trials_temporal, axes=(0, 1))

    #######################  Epoch data class to store the reconstructed ####################
    events = epochs_pos.events
    event_id = epochs_pos.event_id
    tmin = iv_epoch[0]
    sfreq = sfreq

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
    fname = f"{freq_band}_{cond_name}_full.fif"
    bcstp_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    #######################  Epoch data class for the spatially filtered ####################
    events = epochs_pos.events
    event_id = epochs_pos.event_id
    tmin = iv_epoch[0]
    sfreq = sfreq

    ch_names = [f'Component {i + 1}' for i in np.arange(0, keeps)]
    ch_types = ['eeg' for i in np.arange(0, len(ch_names))]

    # Initialize an info structure
    info_full_filt = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    bcstp_epochs_spat = mne.EpochsArray(np.transpose(spatially_filtered_trials, axes=(2, 0, 1)), info_full_filt,
                                        events, tmin,
                                        event_id)
    bcstp_epochs_spat = bcstp_epochs_spat.apply_baseline(baseline=tuple(iv_baseline))

    ################################ Plotting Graphs #######################################
    figure_path_spatial = f'/data/p_02718/Images/bCSTP_eeg/IsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)

    plot_graphs = True
    if plot_graphs:
        ####### Topography for the best spatial patterns ########
        spatial_pat = pattern_from_filter(W[-1][:, 0:4], data_pos)
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axes = plt.subplots(2, 2)
        axes = axes.flatten()
        for icomp in np.arange(0, 4):
            mne.viz.plot_topomap(data=spatial_pat[icomp, :], pos=epochs_pos.info, ch_type='eeg', sensors=True,
                                 contours=6, outlines='head', sphere=None, image_interp='cubic',
                                 extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                 cnorm=None, axes=axes[icomp], show=False)
            axes[icomp].set_title(f'Component {icomp + 1}')
            divider = make_axes_locatable(axes[icomp])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(axes[icomp].images[-1], cax=cax, shrink=0.6, orientation='vertical')
        plt.savefig(figure_path_spatial + f'{freq_band}_{cond_name}.png')
        plt.close(fig)

        ############### Time Course of best Temporal Patterns ###########################
        figure_path_temp_pat = f'/data/p_02718/Images/bCSTP_eeg/TempPatPlots/{subject_id}/'
        os.makedirs(figure_path_temp_pat, exist_ok=True)
        fig, axes = plt.subplots(2, 2)
        for count, ax in enumerate(axes.flatten()):
            data = temporal_patterns[count, :]
            times = epochs_pos.times
            ax.plot(times, data)
            ax.set_title(f'Temporal Pattern {count+1}')
        plt.savefig(figure_path_temp_pat + f'{freq_band}_{cond_name}.png')
        plt.close(fig)

        ################# Spatially Filtered Only #############################
        figure_path_time = f'/data/p_02718/Images/bCSTP_eeg/SpatFiltPlots/{subject_id}/'
        os.makedirs(figure_path_time, exist_ok=True)
        bcstp_evoked = bcstp_epochs_spat.average()

        # Time Course
        fig, axes = plt.subplots(1, 1)
        mne.viz.plot_evoked(bcstp_evoked, picks=None, exclude='bads', unit=True, show=False, ylim=None, xlim='tight',
                            proj=False, hline=None, units=None, scalings=None, titles=None, axes=axes, gfp=False,
                            window_title=None, spatial_colors=False, zorder='unsorted', selectable=True, noise_cov=None,
                            time_unit='s', sphere=None, highlight=None, verbose=None)
        plt.savefig(figure_path_time + f'{freq_band}_{cond_name}_time.png')
        plt.close(fig)

        # Single Trial
        fig, axes = plt.subplots(2, 1)
        vmin = -6e5
        vmax = 6e5
        mne.viz.plot_epochs_image(bcstp_epochs_spat, picks=f'Component 1', sigma=0.0, vmin=vmin, vmax=vmax, colorbar=True,
                                  show=False, units=None, scalings=None, cmap=None, fig=None, axes=axes,
                                  overlay_times=None, combine=None, group_by=None, evoked=False, ts_args=None,
                                  title=f'Component 1', clear=False)
        plt.savefig(figure_path_time + f'{freq_band}_{cond_name}_image.png')
        plt.close(fig)
