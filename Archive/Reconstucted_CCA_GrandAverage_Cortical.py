# Plot grand average time courses and spatial patterns after application of CCA


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_bands = ['sigma', 'kappa']
    srmr_nr = 1
    sfreq = 5000

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG.xlsx')
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    # Get a raw file so I can use the montage
    # raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data/freq_banded_eeg/sub-001/sigma_median.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    # raw.set_montage(montage, on_missing="ignore")
    # eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    # idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    # res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)
    figure_path = '/data/p_02718/Images/Reconstructed_CCA_eeg/GrandAverage/'
    os.makedirs(figure_path, exist_ok=True)

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list = []
            spatial_pattern = []
            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                ##########################################################
                # Time  Course Information
                ##########################################################
                # Select the right files
                fname = f"{freq_band}_{cond_name}.fif"
                input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"

                epochs = mne.read_epochs(input_path + fname, preload=True)

                # Need to pick channel based on excel sheet
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                channel = f'Cor{channel_no}'
                inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                epochs = epochs.pick_channels([channel])
                if inv == 'T':
                    epochs.apply_function(invert, picks=channel)
                # evoked = epochs.average()
                # data = evoked.data
                # evoked_list.append(data)

                ############################################################
                # Spatial Pattern Extraction
                ############################################################
                # Read in saved A_st
                with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                    A_st = pickle.load(f)
                    # Shape (channels, channel_rank)
                if inv == 'T':
                    spatial_pattern.append(A_st[:, channel_no-1]*-1)
                    for_recon = A_st[:, channel_no-1]*-1
                else:
                    spatial_pattern.append(A_st[:, channel_no-1])
                    for_recon = A_st[:, channel_no-1]

                #####################################################################################################
                # Reconstruct the trials in the original electrode domain
                #####################################################################################################
                reconstructed_trials = np.tensordot(epochs.get_data(), for_recon.reshape((len(A_st[:, 0]), 1)),
                                                    axes=(1, 1))
                reconstructed_trials = reconstructed_trials.transpose((0, 2, 1))  # Want (n_epochs, n_channels, n_times)

                ####################################################################################################
                # Get reconstructed trials in an epoch structure
                ####################################################################################################
                events = epochs.events
                event_id = epochs.event_id
                tmin = iv_epoch[0]
                sfreq = sfreq

                ch_names = eeg_chans
                ch_types = ['eeg' for i in np.arange(0, len(ch_names))]

                # Initialize an info structure
                info_full_recon = mne.create_info(
                    ch_names=ch_names,
                    ch_types=ch_types,
                    sfreq=sfreq
                )

                # Create and save
                recon_epochs = mne.EpochsArray(reconstructed_trials, info_full_recon, events, tmin, event_id)
                recon_epochs = recon_epochs.apply_baseline(baseline=tuple(iv_baseline))
                recon_epochs.set_montage(montage, on_missing="ignore")

                # Get correct channels
                if cond_name == 'median':
                    channel = 'CP4'
                    times = [0.016, 0.017, 0.018, 0.019, 0.020, 0.21, 0.022, 0.23, 0.024]

                elif cond_name == 'tibial':
                    channel = 'Cz'
                    times = [0.036, 0.037, 0.038, 0.039, 0.040, 0.041, 0.042, 0.043, 0.044]

                evoked = recon_epochs.average()
                evoked.reorder_channels(eeg_chans)
                evoked_list.append(evoked)

            # Plot time course
            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
            relevant_channel = averaged.copy().pick_channels([channel])
            fig, ax = plt.subplots(1, 1)
            ax.plot(relevant_channel.times, relevant_channel.data[0, :] * 10 ** 6)
            ax.set_ylabel('Amplitude (\u03BCV)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Time Course, n={len(subjects)}')
            if cond_name == 'median':
                ax.set_xlim([0.00, 0.05])
            else:
                ax.set_xlim([0.00, 0.07])
            plt.savefig(figure_path + f'GA_Time_{freq_band}_{cond_name}')

            # Plot Spatial Topographies
            averaged.plot_topomap(times=times, average=None, ch_type=None, scalings=None, proj=False,
                                  sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                  outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                  border='mean',
                                  res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None, cnorm=None,
                                  colorbar=True, cbar_fmt='%3.1f', units=None, axes=None, time_unit='s',
                                  time_format=None, title=f'Grand Average Spatial Pattern, n={len(subjects)}',
                                  nrows=1, ncols='auto', show=True)
            plt.savefig(figure_path + f'GA_Spatial_{freq_band}_{cond_name}')
