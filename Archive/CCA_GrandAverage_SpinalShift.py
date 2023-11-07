# Plot grand average time courses and spatial patterns after application of CCA on ESG data
# Data is shifted to align them based on the latency of the peak of the low-frequency potential
# CAREFUL: For shifting it should be expected - sep_latency


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
    use_updated = True  # If true, use the CCA components from the updated selection
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

    if use_updated:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Updated.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA/GrandAverageShift_Updated/'
        os.makedirs(figure_path, exist_ok=True)
    else:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA/GrandAverageShift/'
        os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

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

                # Only perform if bursts marked as visible
                visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                if visible == 'T':
                    ##########################################################
                    # Time  Course Information
                    ##########################################################
                    # Select the right files
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"

                    epochs = mne.read_epochs(input_path + fname, preload=True)

                    # Need to pick channel based on excel sheet
                    channel_no = int(df.loc[subject, f"{freq_band}_{cond_name}_comp"])
                    channel = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    # data = evoked.data
                    # evoked_list.append(data)

                    # Apply relative time-shift depending on expected latency
                    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
                    fname_pot = 'potential_latency.mat'
                    matdata = loadmat(potential_path + fname_pot)
                    if cond_name == 'median':
                        sep_latency = matdata['med_potlatency']
                        expected = 13 / 1000
                    elif cond_name == 'tibial':
                        sep_latency = matdata['tib_potlatency']
                        expected = 22 / 1000
                    shift = expected - sep_latency[0][0] / 1000
                    evoked.shift_time(shift, relative=True)
                    evoked.crop(tmin=-0.06, tmax=0.20)
                    data = evoked.data
                    evoked_list.append(data)

                    ############################################################
                    # Spatial Pattern Extraction
                    ############################################################
                    # Read in saved A_st
                    with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                        A_st = pickle.load(f)
                        # Shape (channels, channel_rank)
                    if inv == 'T':
                        spatial_pattern.append(A_st[:, channel_no-1]*-1)
                    else:
                        spatial_pattern.append(A_st[:, channel_no-1])

            # Get grand average across chosen epochs, and spatial patterns
            grand_average = np.mean(evoked_list, axis=0)
            grand_average_spatial = np.mean(spatial_pattern, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(evoked.times, grand_average[0, :])
            ax.set_ylabel('Cleaned SEP Amplitude (AU)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Time Course, n={len(evoked_list)}')
            if cond_name == 'median':
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])

            plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}_n={len(evoked_list)}')
            plt.close(fig)

            # Plot Spatial Pattern
            fig, ax = plt.subplots()
            if cond_name == 'median':
                chan_labels = cervical_chans
            elif cond_name == 'tibial':
                chan_labels = lumbar_chans
            if freq_band == 'sigma':
                colorbar_axes = [-0.15, 0.15]
            else:
                colorbar_axes = [-0.01, 0.01]
            subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
            # you can also base the grid on an several subjects
            # then the function takes the average over the channel positions of all those subjects
            time = 0.0
            colorbar = True
            mrmr_esg_isopotentialplot(subjects_4grid, grand_average_spatial, colorbar_axes, chan_labels, colorbar, time,
                                      ax, srmr_nr)
            ax.set_yticklabels([])
            ax.set_ylabel(None)
            ax.set_xticklabels([])
            ax.set_xlabel(None)
            plt.title(f'Grand Average Spatial Pattern, n={len(evoked_list)}')
            plt.savefig(figure_path+f'GA_Spatial_{freq_band}_{cond_name}_n={len(evoked_list)}')
            plt.close(fig)

