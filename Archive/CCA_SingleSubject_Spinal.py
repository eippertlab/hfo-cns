# Plot grid of single subject time domain averages for easy comparison
# Using the selected components after CCA has been performed

import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


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
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
    df_vis = pd.read_excel(xls, 'CCA_Spinal')
    df_vis.set_index('Subject', inplace=True)

    figure_path = '/data/p_02718/Images/CCA/SingleSubject/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for freq_band in freq_bands:
        for condition in conditions:
            fig1, ax1 = plt.subplots(6, 3, figsize=(24, 12))
            ax1 = ax1.flatten()
            fig2, ax2 = plt.subplots(6, 3, figsize=(24, 12))
            ax2 = ax2.flatten()
            count1 = 0
            count2 = 0

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
                evoked = epochs.copy().average()

                # ############################################################
                # # Spatial Pattern Extraction
                # ############################################################
                # # Read in saved A_st
                # with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                #     A_st = pickle.load(f)
                #     # Shape (channels, channel_rank)
                # if inv == 'T':
                #     spatial_pattern = (A_st[:, channel_no - 1] * -1)
                # else:
                #     spatial_pattern = (A_st[:, channel_no - 1])

                # Plot Time Course
                if subject <= 18:
                    ax = ax1[count1]
                    count1 += 1
                else:
                    ax = ax2[count2]
                    count2 += 1
                ax.plot(epochs.times, evoked.get_data().reshape(-1))
                ax.set_ylabel('Amplitude (AU)')
                ax.set_xlabel('Time (s)')
                ax.set_title(f'{subject_id}')
                if cond_name == 'median':
                    ax.set_xlim([0.0, 0.05])
                else:
                    ax.set_xlim([0.0, 0.07])

            fig1.tight_layout()
            fig2.tight_layout()
            fig1.savefig(figure_path + f'1_Time_subplots_{freq_band}_{cond_name}')
            fig2.savefig(figure_path + f'2_Time_subplots_{freq_band}_{cond_name}')
                # plt.savefig(figure_path+f'{subject_id}_Time_{freq_band}_{cond_name}')
                # plt.savefig(figure_path+f'{subject_id}_Time_{freq_band}_{cond_name}.pdf',
                #             bbox_inches='tight', format="pdf")
                # plt.close(fig)

                # # Plot Spatial Pattern
                # fig, ax = plt.subplots()
                # if cond_name == 'median':
                #     chan_labels = cervical_chans
                # elif cond_name == 'tibial':
                #     chan_labels = lumbar_chans
                # if freq_band == 'sigma':
                #     colorbar_axes = [-0.2, 0.2]
                # else:
                #     colorbar_axes = [-0.025, 0.025]
                # subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
                # # you can also base the grid on an several subjects
                # # then the function takes the average over the channel positions of all those subjects
                # time = 0.0
                # colorbar = True
                # mrmr_esg_isopotentialplot(subjects_4grid, spatial_pattern, colorbar_axes, chan_labels, colorbar, time,
                #                           ax)
                # ax.set_yticklabels([])
                # ax.set_ylabel(None)
                # ax.set_xticklabels([])
                # ax.set_xlabel(None)
                # plt.title(f'Spatial Pattern, {subject_id}')
                # plt.savefig(figure_path+f'{subject_id}_Spatial_{freq_band}_{cond_name}')
                # plt.savefig(figure_path+f'{subject_id}_Spatial_{freq_band}_{cond_name}.pdf',
                #             bbox_inches='tight', format="pdf")
                # plt.close(fig)

