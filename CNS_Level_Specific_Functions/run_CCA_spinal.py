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


def run_CCA(subject, condition, srmr_nr, freq_band, freq_type):
    if freq_band != 'sigma':
        raise RuntimeError('Frequency band must be set to sigma, kappa is depreciated')

    plot_graphs = True

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    # Select the right files based on the data_string
    if freq_type == 'high':
        input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
        fname = f"{freq_band}_{cond_name}.fif"
        save_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
        append = ''
    else:
        input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
        fname = f'ssp6_cleaned_{cond_name}.fif'
        save_path = "/data/pt_02718/tmp_data/cca_low/" + subject_id + "/"
        append = '_low'
    os.makedirs(save_path, exist_ok=True)

    figure_path_spatial = f'/data/p_02718/Images/CCA{append}/ComponentIsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)
    figure_path_time = f'/data/p_02718/Images/CCA{append}/ComponentTimePlots/{subject_id}/'
    os.makedirs(figure_path_time, exist_ok=True)
    figure_path = f'/data/p_02718/Images/CCA{append}/ComponentPlots/{subject_id}/'
    os.makedirs(figure_path, exist_ok=True)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    if freq_type == 'low':
        raw.filter(l_freq=1, h_freq=350)

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True)

    if cond_name == 'median':
        epochs = epochs.pick_channels(cervical_chans, ordered=True)
        esg_chans = cervical_chans
        window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccaspinal_med', 'Time'].iloc[0] / 1000,
                        df_timing.loc[df_timing['Name'] == 'tend_ccaspinal_med', 'Time'].iloc[0] / 1000]
        sep_latency = int(df_timing.loc[df_timing['Name'] == 'centre_spinal_med', 'Time'].iloc[0])
    elif cond_name == 'tibial':
        epochs = epochs.pick_channels(lumbar_chans, ordered=True)
        esg_chans = lumbar_chans
        window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccaspinal_tib', 'Time'].iloc[0] / 1000,
                        df_timing.loc[df_timing['Name'] == 'tend_ccaspinal_tib', 'Time'].iloc[0] / 1000]
        sep_latency = int(df_timing.loc[df_timing['Name'] == 'centre_spinal_tib', 'Time'].iloc[0])
    else:
        raise RuntimeError('Invalid condition name attempted for use')

    # Drop bad channels
    # if raw.info['bads']:
    #     for channel in raw.info['bads']:
    #         if channel in esg_chans:
    #             epochs.drop_channels(ch_names=[channel])

    # Crop the epochs
    window = epochs.time_as_index(window_times)
    epo_cca = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data(picks=esg_chans)
    epo_data = epochs.get_data(picks=esg_chans)

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

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ W_st

    # Reshape - (900, 2000, 9)
    no_times_long = np.shape(epochs.get_data())[2]
    no_epochs = np.shape(epochs.get_data())[0]

    # Perform reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    #######################  Epoch data class to store the information ####################
    data = CCA_comps[:, 0:selected_components, :]
    events = epochs.events
    event_id = epochs.event_id
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
    cca_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ################################ Save Spatial Pattern #################################
    afile = open(save_path + f'A_st_{freq_band}_{cond_name}.pkl', 'wb')
    pickle.dump(A_st, afile)
    afile.close()

    # Save correlation coefficients
    rfile = open(save_path + f'r_{freq_band}_{cond_name}.pkl', 'wb')
    pickle.dump(r, rfile)
    rfile.close()

    # Save single trial weights
    rfile = open(save_path + f'W_st_{freq_band}_{cond_name}.pkl', 'wb')
    pickle.dump(W_st, rfile)
    rfile.close()

    ################################ Plotting Graphs #######################################
    if plot_graphs:
        ####### Spinal Isopotential Plots for the first 4 components ########
        # fig, axes = plt.figure()
        fig, axes = plt.subplots(2, 2)
        axes_unflat = axes
        axes = axes.flatten()
        for icomp in np.arange(0, 4):  # Plot for each of four components
            # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
            if freq_band == 'sigma' or freq_band == 'general' or freq_band == 'ktest':
                colorbar_axes = [-0.2, 0.2]
            else:
                colorbar_axes = [-0.025, 0.025]
            chan_labels = epochs.ch_names
            colorbar = True
            time = 0.0
            mrmr_esg_isopotentialplot([subject], A_st[:, icomp], colorbar_axes, chan_labels,
                                      colorbar, time, axes[icomp], srmr_nr)
            axes[icomp].set_title(f'Component {icomp + 1}')
            axes[icomp].set_yticklabels([])
            axes[icomp].set_ylabel(None)
            axes[icomp].set_xticklabels([])
            axes[icomp].set_xlabel(None)

        plt.savefig(figure_path_spatial + f'{freq_band}_{cond_name}.png')
        plt.close(fig)

        ############ Time Course of First 4 components ###############
        fig = plt.figure()
        for icomp in np.arange(0, 4):
            plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
            # Want to plot Cor1 - Cor4
            # Plot for the mixed nerve data
            # get_data returns (n_epochs, n_channels, n_times)
            data = cca_epochs.get_data(picks=[f'Cor{icomp + 1}'])
            to_plot = np.mean(data[:, 0, :], axis=0)
            plt.plot(cca_epochs.times, to_plot)
            plt.xlim([-0.025, 0.065])
            # plt.xlim([0.0, 0.05])
            line_label = f"{sep_latency / 1000}s"
            plt.axvline(x=sep_latency / 1000, color='r', linewidth='0.6', label=line_label)
            plt.xlabel('Time [s]')
            plt.ylabel('Amplitude [A.U.]')
            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_path_time + f'{freq_band}_{cond_name}.png')
        plt.close(fig)

        ############################ Combine to one Image ##########################
        spatial = plt.imread(figure_path_spatial + f'{freq_band}_{cond_name}.png')
        time = plt.imread(figure_path_time + f'{freq_band}_{cond_name}.png')

        fig, axes = plt.subplots(1, 2, figsize=(10, 6))
        axes[0].imshow(time)
        axes[0].axis('off')
        axes[1].imshow(spatial)
        axes[1].axis('off')

        plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

        plt.suptitle(f'Subject {subject}, {freq_band}_{cond_name}')
        plt.savefig(figure_path + f'{freq_band}_{cond_name}.png')
        plt.close(fig)

