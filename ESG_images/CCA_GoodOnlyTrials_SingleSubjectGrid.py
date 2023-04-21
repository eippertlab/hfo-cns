# Plot grand average time courses and envelope of CCA on ESG data
# Single subject grid image
# Can also plot spatial topography in similar grid fashion


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
import pickle


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_bands = ['sigma']
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
    df = pd.read_excel(xls, 'CCA_goodonly')
    df.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
    df_vis = pd.read_excel(xls, 'CCA_Spinal_GoodOnly')
    df_vis.set_index('Subject', inplace=True)

    figure_path = '/data/p_02718/Images/CCA_good/SingleSubject/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    time = True  # If time is true, plot evoked & envelope
    # Otherwise, plot the spatial patterns

    for freq_band in freq_bands:
        for condition in conditions:
            fig, axes = plt.subplots(6, 6, figsize=(12, 12))
            axes = axes.flatten()
            for n, subject in enumerate(subjects):
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Check if bursts are marked as visible
                visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]

                # Select the right files
                fname = f"{freq_band}_{cond_name}.fif"
                input_path = "/data/pt_02718/tmp_data/cca_goodonly/" + subject_id + "/"

                # Need to pick channel based on excel sheet
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                channel = f'Cor{channel_no}'
                inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]

                if time is True:
                    ##########################################################
                    # Time  Course Information
                    ##########################################################
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    data_evoked = evoked.get_data()

                    ##############################################################
                    # Envelope
                    ##############################################################
                    envelope = evoked.apply_hilbert(envelope=True)
                    data_envelope = envelope.get_data()

                    axes[n].plot(evoked.times, data_evoked.reshape(-1))
                    if visible == 'T':
                        color = 'green'
                    else:
                        color = 'red'
                    axes[n].plot(evoked.times, data_envelope.reshape(-1), color=color)
                    # axes[n].set_ylabel('Cleaned SEP Amplitude (AU)')
                    # axes[n].set_xlabel('Time (s)')
                    axes[n].set_title(f'Subject {n + 1}')
                    axes[n].set_xlim([0.0, 0.05])

                    ###############################################################
                    # Add line for expected latency
                    ###############################################################
                    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
                    fname_pot = 'potential_latency.mat'
                    matdata = loadmat(potential_path + fname_pot)
                    if cond_name == 'median':
                        sep_latency = matdata['med_potlatency']
                    elif cond_name == 'tibial':
                        sep_latency = matdata['tib_potlatency']
                    axes[n].axvline(x=sep_latency[0][0] / 1000, color='r', linewidth='1')
                    plt.tight_layout()

                else:
                    ############################################################
                    # Spatial Pattern Extraction
                    ############################################################
                    # Read in saved A_st
                    with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                        A_st = pickle.load(f)
                        # Shape (channels, channel_rank)
                    if inv == 'T':
                        spatial_pattern = A_st[:, channel_no-1]*-1
                    else:
                        spatial_pattern = A_st[:, channel_no-1]
                    if cond_name == 'median':
                        chan_labels = cervical_chans
                    elif cond_name == 'tibial':
                        chan_labels = lumbar_chans
                    if freq_band == 'sigma':
                        colorbar_axes = [-0.2, 0.2]
                    else:
                        colorbar_axes = [-0.01, 0.01]
                    subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
                    # you can also base the grid on an several subjects
                    # then the function takes the average over the channel positions of all those subjects
                    time = 0.0
                    colorbar = True
                    mrmr_esg_isopotentialplot(subjects_4grid, spatial_pattern, colorbar_axes, chan_labels,
                                              colorbar, time, axes[n])
                    axes[n].set_yticklabels([])
                    axes[n].set_ylabel(None)
                    axes[n].set_xticklabels([])
                    axes[n].set_xlabel(None)
                    axes[n].set_title(f'Subject {n+1}')

            if time is True:
                plt.savefig(figure_path+f'{cond_name}.png')
            else:
                plt.tight_layout()
                plt.savefig(figure_path+f'{cond_name}_spatial.png')

    plt.show()

