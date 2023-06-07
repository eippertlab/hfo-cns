# Script to actually run CCA on the data from dataset 2
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA
# As with the spinal data, we train with the mixed nerve data and apply the weights to the digit data


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def run_CCA2(subject, condition, srmr_nr, freq_band, sfreq):
    # Read in locations of channels
    plot_graphs = True

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name_dig = cond_info.cond_name
    trigger_name_dig = list(cond_info.trigger_name)
    # trigger_name = trigger_name_dig.copy()
    cond_info = get_conditioninfo(condition + 1, srmr_nr)
    cond_name_mixed = cond_info.cond_name
    trigger_name_mixed = cond_info.trigger_name
    # trigger_name.append(trigger_name_mixed)
    subject_id = f'sub-{str(subject).zfill(3)}'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    # Select the right files based on the data_string
    input_path = "/data/pt_02718/tmp_data_2/freq_banded_eeg/" + subject_id + "/"
    fname_dig = f"{freq_band}_{cond_name_dig}.fif"
    fname_mixed = f"{freq_band}_{cond_name_mixed}.fif"
    save_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

    raw_dig = mne.io.read_raw_fif(input_path + fname_dig, preload=True)
    raw_mixed = mne.io.read_raw_fif(input_path + fname_mixed, preload=True)

    # Set montage
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    for raw in [raw_dig, raw_mixed]:
        raw.set_montage(montage, on_missing="ignore")
        # idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)

    # now create epochs based on the trigger names
    # Mixed data
    events, event_ids = mne.events_from_annotations(raw_mixed)
    event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_name_mixed}
    epochs_m = mne.Epochs(raw_mixed, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                          baseline=tuple(iv_baseline), preload=True)

    # Digits data
    events_d, event_ids_d = mne.events_from_annotations(raw_dig)
    event_id_dict_d = {key: value for key, value in event_ids_d.items() if key in trigger_name_dig}
    epochs_d = mne.Epochs(raw_dig, events_d, event_id=event_id_dict_d, tmin=iv_epoch[0],
                          tmax=iv_epoch[1] - 1 / 1000,
                          baseline=tuple(iv_baseline), preload=True)

    if 'med1' in trigger_name_dig:  # Using list of triggers to check if this is tibial/median
        epochs_d = epochs_d.pick_channels(eeg_chans, ordered=True)
        epochs_m = epochs_m.pick_channels(eeg_chans, ordered=True)
        if freq_band == 'sigma':
            window_times = [15.4 / 1000, 24.8 / 1000]
        elif freq_band == 'kappa':
            window_times = [18.4 / 1000, 24.8 / 1000]
        sep_latency = 20
    elif 'tib1' in trigger_name_dig:
        epochs_d = epochs_d.pick_channels(eeg_chans, ordered=True)
        epochs_m = epochs_m.pick_channels(eeg_chans, ordered=True)
        if freq_band == 'sigma':
            window_times = [32 / 1000, 44 / 1000]
        elif freq_band == 'kappa':
            window_times = [32 / 1000, 44 / 1000]
        sep_latency = 40
    else:
        print('Invalid condition name attempted for use')
        exit()

    # Drop bad channels  -  if a channel is bad in mixed, drop it from mixed & digits
    # if raw_mixed.info['bads'] or raw_dig.info['bads']:
    #     for channel in raw_mixed.info['bads'] + raw_dig.info['bads']:
    #         if channel in eeg_chans:
    #             epochs_d.drop_channels(ch_names=[channel])
    #             epochs_m.drop_channels(ch_names=[channel])
    # if raw_mixed.info['bads']:
    #     for channel in raw_mixed.info['bads']:
    #         if channel in esg_chans:
    #             # epochs_d.drop_channels(ch_names=[channel])
    #             epochs_m.drop_channels(ch_names=[channel])

    # We plot for the mixed data so we want this below - need info object after channels dropped
    idx_by_type = mne.channel_indices_by_type(epochs_m.info, picks=eeg_chans)
    res = mne.pick_info(epochs_m.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

    # Crop the epochs
    window = epochs_m.time_as_index(window_times)
    epo_cca = epochs_m.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs_m.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data(picks=eeg_chans)
    epo_data = epochs_m.get_data(picks=eeg_chans)

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    no_times = int(window[1] - window[0])
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T

    # Run CCA
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)

    all_components = len(r)

    # Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
    CCA_concat = st_matrix_long.T @ W_st[:, 0:all_components]
    CCA_concat = CCA_concat.T

    # Need to get digits matrix
    epo_data_d = epochs_d.get_data(picks=eeg_chans)
    st_matrix_digits = np.swapaxes(epo_data_d, 1, 2).reshape(-1, epo_data_d.shape[1]).T
    CCA_concat_d = st_matrix_digits.T @ W_st[:, 0:all_components]
    CCA_concat_d = CCA_concat_d.T

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Reshape - (900, 2000, 9)
    no_times_long = np.shape(epochs_m.get_data())[2]
    no_epochs = np.shape(epochs_m.get_data())[0]
    no_times_long_d = np.shape(epochs_d.get_data())[2]
    no_epochs_d = np.shape(epochs_d.get_data())[0]

    # Perform reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')
    CCA_comps_d = np.reshape(CCA_concat_d, (all_components, no_times_long_d, no_epochs_d), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)

    CCA_comps_d = np.swapaxes(CCA_comps_d, 0, 2)
    CCA_comps_d = np.swapaxes(CCA_comps_d, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    #######################  Epoch data class to store the information ####################
    data = CCA_comps[:, 0:selected_components, :]
    events = epochs_m.events
    event_id = epochs_m.event_id
    tmin = iv_epoch[0]
    sfreq = sfreq

    ch_names = []
    ch_types = []
    for i in np.arange(0, all_components):
        ch_names.append(f'Cor{i+1}')
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    cca_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    cca_epochs = cca_epochs.apply_baseline(baseline=tuple(iv_baseline))
    cca_epochs.save(os.path.join(save_path, fname_mixed), fmt='double', overwrite=True)

    # Digits data
    data_d = CCA_comps_d[:, 0:selected_components, :]
    events_d = epochs_d.events
    event_id_d = epochs_d.event_id

    # Create and save
    cca_epochs_d = mne.EpochsArray(data_d, info, events_d, tmin, event_id_d)
    cca_epochs_d = cca_epochs_d.apply_baseline(baseline=tuple(iv_baseline))
    # cca_epochs['med1'].average(picks='Cor1').plot()  # Just testing the data to see if it's prettier
    cca_epochs_d.save(os.path.join(save_path, fname_dig), fmt='double', overwrite=True)

    ################################ Save Spatial Pattern #################################
    afile = open(save_path + f'A_st_{freq_band}_{cond_name_mixed}.pkl', 'wb')
    pickle.dump(A_st, afile)
    afile.close()

    ################################ Plotting Graphs #######################################
    figure_path_spatial = f'/data/p_02718/Images_2/CCA_eeg/ComponentIsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)

    if plot_graphs:
        ####### Isopotential Plots for the first 4 components ########
        # fig, axes = plt.figure()
        fig, axes = plt.subplots(2, 2)
        axes_unflat = axes
        axes = axes.flatten()
        for icomp in np.arange(0, 4):  # Plot for each of four components
            # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
            chan_labels = epochs_m.ch_names
            mne.viz.plot_topomap(data=A_st[:, icomp], pos=res, ch_type='eeg', sensors=True, names=None,
                                 contours=6, outlines='head', sphere=None, image_interp='cubic',
                                 extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                 cnorm=None, axes=axes[icomp], show=False)
            axes[icomp].set_title(f'Component {icomp + 1}')
            divider = make_axes_locatable(axes[icomp])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(axes[icomp].images[-1], cax=cax, shrink=0.6, orientation='vertical')

        plt.savefig(figure_path_spatial + f'{freq_band}_{cond_name_mixed}.png')
        plt.close(fig)

        ############ Time Course of First 4 components ###############
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        figure_path_time = f'/data/p_02718/Images_2/CCA_eeg/ComponentTimePlots/{subject_id}/'
        os.makedirs(figure_path_time, exist_ok=True)

        fig = plt.figure()
        for icomp in np.arange(0, 4):
            plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
            # Want to plot Cor1 - Cor4
            # Plot for the mixed nerve data
            # get_data returns (n_epochs, n_channels, n_times)
            data = cca_epochs.get_data(picks=[f'Cor{icomp + 1}'])
            to_plot = np.mean(data[:, 0, :], axis=0)
            plt.plot(cca_epochs.times, to_plot)
            plt.xlim([0.01, 0.07])
            line_label = f"{sep_latency / 1000}s"
            plt.axvline(x=sep_latency / 1000, color='r', linewidth='0.6', label=line_label)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [A.U.]')
            plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path_time + f'{freq_band}_{cond_name_mixed}.png')
        plt.close(fig)

        ############################ Combine to one Image ##########################
        figure_path = f'/data/p_02718/Images_2/CCA_eeg/ComponentPlots/{subject_id}/'
        os.makedirs(figure_path, exist_ok=True)

        spatial = plt.imread(figure_path_spatial + f'{freq_band}_{cond_name_mixed}.png')
        time = plt.imread(figure_path_time + f'{freq_band}_{cond_name_mixed}.png')
        # single_trial = plt.imread(figure_path_st + f'{freq_band}_{cond_name}.png')

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(time)
        axes[0].axis('off')
        axes[1].imshow(spatial)
        axes[1].axis('off')
        # axes[1, 0].imshow(single_trial)
        # axes[1, 0].axis('off')
        # axes[1, 1].axis('off')

        plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

        plt.suptitle(f'Subject {subject}, {freq_band}_{cond_name_mixed}')
        plt.savefig(figure_path + f'{freq_band}_{cond_name_mixed}.png')
        plt.close(fig)

