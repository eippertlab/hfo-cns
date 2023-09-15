# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA
# This script only works if we rerun the bad_trial check after SSP to avoid filtering the raw we resave between
# 400 and 1400Hz


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


def run_CCA_highlow(subject, condition, srmr_nr, freq_band):
    plot_graphs = True

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
    input_path_high = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
    fname_high = f"{freq_band}_{cond_name}.fif"
    input_path_low = f"/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
    fname_low = f"ssp6_cleaned_{cond_name}.fif"
    save_path = "/data/pt_02718/tmp_data/cca_highlow/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23', 'TH6']

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    raw_low = mne.io.read_raw_fif(input_path_low + fname_low, preload=True)
    raw_high = mne.io.read_raw_fif(input_path_high + fname_high, preload=True)

    # now create epochs based on the trigger names
    # Low frequency data
    events, event_ids = mne.events_from_annotations(raw_low)
    event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_name}
    epochs_l = mne.Epochs(raw_low, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                          baseline=tuple(iv_baseline), preload=True)

    # High frequency data
    events_h, event_ids_h = mne.events_from_annotations(raw_high)
    event_id_dict_h = {key: value for key, value in event_ids_h.items() if key in trigger_name}
    epochs_h = mne.Epochs(raw_high, events_h, event_id=event_id_dict_h, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                          baseline=tuple(iv_baseline), preload=True)

    # cca window size - Birgit created individual potential latencies for each subject
    fname_pot = 'potential_latency.mat'
    matdata = loadmat(potential_path + fname_pot)

    if cond_name == 'median':
        epochs_h = epochs_h.pick_channels(cervical_chans, ordered=True)
        epochs_l = epochs_l.pick_channels(cervical_chans, ordered=True)
        esg_chans = cervical_chans
        sep_latency = matdata['med_potlatency']
        # window_times = [7/1000, 37/1000]
        window_times = [7/1000, 22/1000]
    elif cond_name == 'tibial':
        epochs_h = epochs_h.pick_channels(lumbar_chans, ordered=True)
        epochs_l = epochs_l.pick_channels(lumbar_chans, ordered=True)
        esg_chans = lumbar_chans
        sep_latency = matdata['tib_potlatency']
        # window_times = [7/1000, 47/1000]
        window_times = [15/1000, 30/1000]
    else:
        print('Invalid condition name attempted for use')
        exit()

    # Drop bad channels
    # if raw.info['bads']:
    #     for channel in raw.info['bads']:
    #         if channel in esg_chans:
    #             epochs.drop_channels(ch_names=[channel])

    # Crop the epochs
    window = epochs_l.time_as_index(window_times)
    epo_cca = epochs_l.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs_l.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data(picks=esg_chans)
    epo_data = epochs_l.get_data(picks=esg_chans)

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

    # Need to get high frequency matrix
    epo_data_h = epochs_h.get_data(picks=esg_chans)
    st_matrix_high = np.swapaxes(epo_data_h, 1, 2).reshape(-1, epo_data_h.shape[1]).T
    CCA_concat_h = st_matrix_high.T @ W_st[:, 0:all_components]
    CCA_concat_h = CCA_concat_h.T

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st
    A_st_h = np.cov(st_matrix_high) @ W_st

    # Reshape - (900, 2000, 9)
    no_times_long = np.shape(epochs_l.get_data())[2]
    no_epochs = np.shape(epochs_l.get_data())[0]
    no_times_long_h = np.shape(epochs_h.get_data())[2]
    no_epochs_h = np.shape(epochs_h.get_data())[0]

    # Perform reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')
    CCA_comps_h = np.reshape(CCA_concat_h, (all_components, no_times_long_h, no_epochs_h), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)

    CCA_comps_h = np.swapaxes(CCA_comps_h, 0, 2)
    CCA_comps_h = np.swapaxes(CCA_comps_h, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    #######################  Epoch data class to store the information ####################
    data = CCA_comps[:, 0:selected_components, :]
    events = epochs_l.events
    event_id = epochs_l.event_id
    tmin = iv_epoch[0]
    sfreq = 5000

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
    cca_epochs.save(os.path.join(save_path, fname_low), fmt='double', overwrite=True)

    # High frequency data
    data_h = CCA_comps_h[:, 0:selected_components, :]
    events_h = epochs_h.events
    event_id_h = epochs_h.event_id

    # Create and save
    cca_epochs_h = mne.EpochsArray(data_h, info, events_h, tmin, event_id_h)
    cca_epochs_h = cca_epochs_h.apply_baseline(baseline=tuple(iv_baseline))
    # cca_epochs['med1'].average(picks='Cor1').plot()  # Just testing the data to see if it's prettier
    cca_epochs_h.save(os.path.join(save_path, fname_high), fmt='double', overwrite=True)

    ################################ Save Spatial Pattern #################################
    afile = open(save_path + f'A_st_{cond_name}_low.pkl', 'wb')
    pickle.dump(A_st, afile)
    afile.close()

    afile = open(save_path + f'A_st_{cond_name}_high.pkl', 'wb')
    pickle.dump(A_st_h, afile)
    afile.close()

    ################################ Plotting Graphs #######################################
    figure_path_spatial = f'/data/p_02718/Images/CCA_highlow/ComponentIsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)

    if plot_graphs:
        ####### Spinal Isopotential Plots for the first 4 components for both high and low frequency ########
        # fig, axes = plt.figure()
        for freq in ['low', 'high']:
            if freq == 'low':
                names = epochs_l.ch_names
                pattern = A_st
            elif freq == 'high':
                names = epochs_h.ch_names
                pattern = A_st_h
            fig, axes = plt.subplots(2, 2)
            axes_unflat = axes
            axes = axes.flatten()
            for icomp in np.arange(0, 4):  # Plot for each of four components
                # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
                colorbar_axes = [-0.2, 0.2]
                chan_labels = names
                colorbar = True
                time = 0.0
                mrmr_esg_isopotentialplot([subject], pattern[:, icomp], colorbar_axes, chan_labels,
                                          colorbar, time, axes[icomp])
                axes[icomp].set_title(f'Component {icomp + 1}')
                axes[icomp].set_yticklabels([])
                axes[icomp].set_ylabel(None)
                axes[icomp].set_xticklabels([])
                axes[icomp].set_xlabel(None)

            plt.savefig(figure_path_spatial + f'{cond_name}_{freq}.png')
            plt.close(fig)

        ############ Time Course of First 4 components ###############
        # cca_epochs and cca_epochs_d both already baseline corrected before this point
        figure_path_time = f'/data/p_02718/Images/CCA_highlow/ComponentTimePlots/{subject_id}/'
        os.makedirs(figure_path_time, exist_ok=True)

        for freq in ['low', 'high']:
            if freq == 'low':
                cca_toplot = cca_epochs
            elif freq == 'high':
                cca_toplot = cca_epochs_h
            fig = plt.figure()
            for icomp in np.arange(0, 4):
                plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
                # Want to plot Cor1 - Cor4
                # Plot for the mixed nerve data
                # get_data returns (n_epochs, n_channels, n_times)
                data = cca_toplot.get_data(picks=[f'Cor{icomp + 1}'])
                to_plot = np.mean(data[:, 0, :], axis=0)
                plt.plot(cca_epochs.times, to_plot)
                plt.xlim([-0.025, 0.065])
                line_label = f"{sep_latency[0][0] / 1000}s"
                plt.axvline(x=sep_latency[0][0] / 1000, color='r', linewidth='0.6', label=line_label)
                plt.xlabel('Time [s]')
                plt.ylabel('Amplitude [A.U.]')
                plt.legend()
                plt.tight_layout()

            plt.savefig(figure_path_time + f'{cond_name}_{freq}.png')
            plt.close(fig)

        ############################ Combine to one Image ##########################
        figure_path = f'/data/p_02718/Images/CCA_highlow/ComponentPlots/{subject_id}/'
        os.makedirs(figure_path, exist_ok=True)

        for freq in ['high', 'low']:
            spatial = plt.imread(figure_path_spatial + f'{cond_name}_{freq}.png')
            time = plt.imread(figure_path_time + f'{cond_name}_{freq}.png')

            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            axes[0].imshow(time)
            axes[0].axis('off')
            axes[1].imshow(spatial)
            axes[1].axis('off')

            plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

            plt.suptitle(f'Subject {subject}, {freq}_{cond_name}')
            plt.savefig(figure_path + f'{cond_name}_{freq}.png')
            plt.close(fig)