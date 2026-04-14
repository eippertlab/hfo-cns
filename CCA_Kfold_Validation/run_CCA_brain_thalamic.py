# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from sklearn.model_selection import KFold
from CCA_Kfold_Validation.apply_cca import apply_cca
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import random
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


def run_CCA_thalamic(subject, condition, srmr_nr, freq_band, sfreq, freq_type, k):
    if freq_band != 'sigma':
        raise RuntimeError('Frequency band must be set to sigma, kappa is depreciated')

    # Read in locations of channels
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
        input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
        fname = f"{freq_band}_{cond_name}.fif"
        save_path = "/data/pt_02718/tmp_data/cca_kfold_eeg_thalamic/" + subject_id + "/"
        append = ''
    else:
        input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
        fname = f'noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif'
        save_path = "/data/pt_02718/tmp_data/cca_kfold_eeg_thalamic_low/" + subject_id + "/"
        append = '_low'
    os.makedirs(save_path, exist_ok=True)

    figure_path_spatial = f'/data/p_02718/Images/CCA_Kfold_eeg_thalamic{append}/ComponentIsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)
    figure_path_time = f'/data/p_02718/Images/CCA_Kfold_eeg_thalamic{append}/ComponentTimePlots/{subject_id}/'
    os.makedirs(figure_path_time, exist_ok=True)
    figure_path = f'/data/p_02718/Images/CCA_Kfold_eeg_thalamic{append}/ComponentPlots/{subject_id}/'
    os.makedirs(figure_path, exist_ok=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    if freq_type == 'low':
        raw.filter(l_freq=1, h_freq=350)

    # Set montage
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                        baseline=tuple(iv_baseline), preload=True, reject_by_annotation=False)

    if cond_name == 'median':
        epochs = epochs.pick_channels(eeg_chans, ordered=True)
        window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccasub_med', 'Time'].iloc[0] / 1000,
                        df_timing.loc[df_timing['Name'] == 'tend_ccasub_med', 'Time'].iloc[0] / 1000]
        sep_latency = int(df_timing.loc[df_timing['Name'] == 'centre_sub_med', 'Time'].iloc[0])
    elif cond_name == 'tibial':
        epochs = epochs.pick_channels(eeg_chans, ordered=True)
        window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccasub_tib', 'Time'].iloc[0] / 1000,
                        df_timing.loc[df_timing['Name'] == 'tend_ccasub_tib', 'Time'].iloc[0] / 1000]
        sep_latency = int(df_timing.loc[df_timing['Name'] == 'centre_sub_tib', 'Time'].iloc[0])
    else:
        raise RuntimeError('Invalid condition name attempted for use')

    # For plotting of spatial topographies later
    idx_by_type = mne.channel_indices_by_type(epochs.info, picks=eeg_chans)
    res = mne.pick_info(epochs.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

    # Crop the epochs for CCA and subselect 50% for training and 50% for application from both full and cropped epochs
    window = epochs.time_as_index(window_times)
    kf = KFold(n_splits=k, shuffle=True, random_state=321)
    kf.get_n_splits()
    for k_ind, (train_index, test_index) in enumerate(kf.split(range(0, 1999))):
        epochs_train = epochs.copy()[train_index]
        epo_cca_train = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)[train_index]
        epochs_apply = epochs.copy()[test_index]
        epo_cca_apply = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)[test_index]

        CCA_comps, A_st, W_st, r, all_components = apply_cca(epochs_train, epochs_apply, epo_cca_train, epo_cca_apply,
                                                             eeg_chans, window)

        #######################  Epoch data class to store the information ####################
        data = CCA_comps[:, 0:all_components, :]
        events = epochs_apply.events
        event_id = epochs_apply.event_id
        tmin = iv_epoch[0]
        sfreq = 5000

        ch_names = []
        ch_types = []
        for i in np.arange(0, all_components):
            ch_names.append(f'Cor{i + 1}')
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
        cca_epochs.save(os.path.join(save_path, f"{k_ind}fold_{fname}"), fmt='double', overwrite=True)

        ################################ Save Spatial Pattern #################################
        afile = open(save_path + f'{k_ind}fold_A_st_{freq_band}_{cond_name}.pkl', 'wb')
        pickle.dump(A_st, afile)
        afile.close()

        # Save correlation coefficients
        rfile = open(save_path + f'{k_ind}fold_r_{freq_band}_{cond_name}.pkl', 'wb')
        pickle.dump(r, rfile)
        rfile.close()

        # Save single trial weights
        rfile = open(save_path + f'{k_ind}fold_W_st_{freq_band}_{cond_name}.pkl', 'wb')
        pickle.dump(W_st, rfile)
        rfile.close()

        ################################ Plotting Graphs #######################################
        if plot_graphs:
            ####### Isopotential Plots for the first 4 components ########
            # fig, axes = plt.figure()
            fig, axes = plt.subplots(2, 2)
            axes_unflat = axes
            axes = axes.flatten()
            for icomp in np.arange(0, 4):  # Plot for each of four components
                # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
                chan_labels = epochs.ch_names
                mne.viz.plot_topomap(data=A_st[:, icomp], pos=res, ch_type='eeg', sensors=True, names=None,
                                     contours=6, outlines='head', sphere=None, image_interp='cubic',
                                     extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                     cnorm=None, axes=axes[icomp], show=False)
                axes[icomp].set_title(f'Component {icomp + 1}')
                divider = make_axes_locatable(axes[icomp])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = fig.colorbar(axes[icomp].images[-1], cax=cax, shrink=0.6, orientation='vertical')

            plt.savefig(figure_path_spatial + f'{k_ind}fold_{freq_band}_{cond_name}.png')
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
                plt.xlim([0.01, 0.07])
                line_label = f"{sep_latency / 1000}s"
                plt.axvline(x=sep_latency / 1000, color='r', linewidth='0.6', label=line_label)
                plt.xlabel('Time [s]')
                plt.ylabel('Amplitude [A.U.]')
                plt.legend()
            plt.tight_layout()
            plt.savefig(figure_path_time + f'{k_ind}fold_{freq_band}_{cond_name}.png')
            plt.close(fig)

            ############################ Combine to one Image ##########################
            spatial = plt.imread(figure_path_spatial + f'{k_ind}fold_{freq_band}_{cond_name}.png')
            time = plt.imread(figure_path_time + f'{k_ind}fold_{freq_band}_{cond_name}.png')

            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            axes[0].imshow(time)
            axes[0].axis('off')
            axes[1].imshow(spatial)
            axes[1].axis('off')

            plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

            plt.suptitle(f'Subject {subject}, {freq_band}_{cond_name}')
            plt.savefig(figure_path + f'{k_ind}fold_{freq_band}_{cond_name}.png')
            plt.close(fig)
