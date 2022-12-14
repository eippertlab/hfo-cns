# Plot envelope with hilbert transform


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.get_esg_channels import get_esg_channels
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot


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
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    figure_path = '/data/p_02718/Images/Reconstructed_CCA/Envelope/'
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
                input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"

                epochs = mne.read_epochs(input_path + fname, preload=True)

                # Need to pick channel based on excel sheet
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                channel = f'Cor{channel_no}'
                inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                epochs = epochs.pick_channels([channel])
                if inv == 'T':
                    epochs.apply_function(invert, picks=channel)

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

                if cond_name == 'median':
                    ch_names = cervical_chans
                elif cond_name == 'tibial':
                    ch_names = lumbar_chans
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

                # Get correct channels
                if cond_name == 'median':
                    channel = 'SC6'
                    times = [0.009, 0.011, 0.013, 0.015, 0.017]

                elif cond_name == 'tibial':
                    channel = 'L1'
                    times = [0.019, 0.021, 0.023, 0.025, 0.027]

                recon_epochs = recon_epochs.pick_channels([channel])
                evoked = recon_epochs.average()
                envelope = evoked.apply_hilbert(envelope=True)
                data = envelope.get_data()
                evoked_list.append(data)

            # Plot time course
            grand_average = np.mean(evoked_list, axis=0)
            fig, ax = plt.subplots(1, 1)
            ax.plot(epochs.times, grand_average[0, :])
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Envelope, n={len(subjects)}')
            if cond_name == 'median':
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])
            plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}')
