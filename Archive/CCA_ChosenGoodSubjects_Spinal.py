# Plot grand average time courses and spatial patterns after application of CCA on ESG data
# Only for a few select subjects that yield good images - for posters


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    subjects = [6, 12, 16, 18, 22, 31, 36]
    conditions = [2, 3]
    freq_bands = ['sigma']
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    use_only_good = True  # From CCA when run using only the pre-screened good trials

    if use_only_good:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
        df = pd.read_excel(xls, 'CCA_goodonly')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal_GoodOnly')
        df_vis.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA_good/GoodSubjects/'
        os.makedirs(figure_path, exist_ok=True)
    else:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA/GoodSubjects/'
        os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for freq_band in freq_bands:
        for condition in conditions:
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
                if use_only_good:
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = "/data/pt_02718/tmp_data/cca_goodonly/" + subject_id + "/"
                else:
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

                ############################################################
                # Spatial Pattern Extraction
                ############################################################
                # Read in saved A_st
                with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                    A_st = pickle.load(f)
                    # Shape (channels, channel_rank)
                if inv == 'T':
                    spatial_pattern = (A_st[:, channel_no - 1] * -1)
                else:
                    spatial_pattern = (A_st[:, channel_no - 1])

                # Plot Time Course
                fig, ax = plt.subplots()
                ax.plot(epochs.times, evoked.get_data().reshape(-1))
                ax.set_ylabel('Cleaned SEP Amplitude (AU)')
                ax.set_xlabel('Time (s)')
                ax.set_title(f'Time Course, {subject_id}')
                if cond_name == 'median':
                    ax.set_xlim([0.0, 0.05])
                else:
                    ax.set_xlim([0.0, 0.07])

                plt.savefig(figure_path+f'{subject_id}_Time_{freq_band}_{cond_name}')
                plt.savefig(figure_path+f'{subject_id}_Time_{freq_band}_{cond_name}.pdf',
                            bbox_inches='tight', format="pdf")
                plt.close(fig)

                # Plot Spatial Pattern
                fig, ax = plt.subplots()
                if cond_name == 'median':
                    chan_labels = cervical_chans
                elif cond_name == 'tibial':
                    chan_labels = lumbar_chans
                if freq_band == 'sigma':
                    colorbar_axes = [-0.2, 0.2]
                else:
                    colorbar_axes = [-0.025, 0.025]
                subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
                # you can also base the grid on an several subjects
                # then the function takes the average over the channel positions of all those subjects
                time = 0.0
                colorbar = True
                mrmr_esg_isopotentialplot(subjects_4grid, spatial_pattern, colorbar_axes, chan_labels, colorbar, time,
                                          ax, srmr_nr)
                ax.set_yticklabels([])
                ax.set_ylabel(None)
                ax.set_xticklabels([])
                ax.set_xlabel(None)
                plt.title(f'Spatial Pattern, {subject_id}')
                plt.savefig(figure_path+f'{subject_id}_Spatial_{freq_band}_{cond_name}')
                plt.savefig(figure_path+f'{subject_id}_Spatial_{freq_band}_{cond_name}.pdf',
                            bbox_inches='tight', format="pdf")
                plt.close(fig)

