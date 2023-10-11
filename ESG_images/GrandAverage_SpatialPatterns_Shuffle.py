# Plot grand average spatial patterns after application of CCA for HFOs
# Plot grand average spatial patterns for low frequency raw potentials


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
import matplotlib.pyplot as plt
from Common_Functions.IsopotentialFunctions_CbarLabel import mrmr_esg_isopotentialplot
import matplotlib as mpl
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    use_visible = True  # Use only subjects with visible bursting

    subjects = np.arange(1, 37)
    conditions = [2]  # No surviving subjects for tibial so no need to run
    freq_bands = ['sigma']
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    figure_path = '/data/p_02718/Images/CCA_shuffle/GrandAverage_SpatialPatterns/'
    os.makedirs(figure_path, exist_ok=True)

    # Spinal Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Shuffle_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Shuffle_Updated.xlsx')
    df_vis_spinal = pd.read_excel(xls, 'CCA_Spinal')
    df_vis_spinal.set_index('Subject', inplace=True)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)

    for freq_band in freq_bands:
        for condition in conditions:
            spatial_pattern = []
            data_list = []
            evoked_list = []

            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Select the right files
                # HFO
                fname = f"{freq_band}_{cond_name}.fif"
                input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                df = df_spinal
                df_vis = df_vis_spinal

                # Low Freq SEP
                input_path_low = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                fname_low = f"epochs_{cond_name}.fif"
                epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                evoked_low = epochs_low.average()

                if trigger_name == 'Median - Stimulation':
                    time_point = 13 / 1000
                    channel = ['SC6']
                else:
                    time_point = 22 / 1000
                    channel = ['L1']

                # Need to pick channel based on excel sheet
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                channel = f'Cor{channel_no}'
                inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]

                ###########################################################
                # Spatial Pattern Extraction for HFOs
                ############################################################
                # Read in saved A_st
                with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                    A_st = pickle.load(f)
                    # Shape (channels, channel_rank)
                if use_visible is True:
                    visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                    if visible == 'T':
                        if inv == 'T':
                            spatial_pattern.append(A_st[:, channel_no - 1] * -1)
                        else:
                            spatial_pattern.append(A_st[:, channel_no - 1])
                        evoked_low.crop(tmin=time_point, tmax=time_point + (2 / 1000))
                        data = evoked_low.data.mean(axis=1)
                        data_list.append(data)
                else:
                    if inv == 'T':
                        spatial_pattern.append(A_st[:, channel_no - 1] * -1)
                    else:
                        spatial_pattern.append(A_st[:, channel_no - 1])

                    evoked_low.crop(tmin=time_point, tmax=time_point + (2 / 1000))
                    data = evoked_low.data.mean(axis=1)
                    data_list.append(data)

            # Get grand average
            grand_average_spatial = np.mean(spatial_pattern, axis=0)  # HFO

            ##########################################################################################
            # HFO
            ##########################################################################################
            fig, ax = plt.subplots()
            if cond_name == 'median':
                chan_labels = cervical_chans
                colorbar_axes = [-0.15, 0.15]
            elif cond_name == 'tibial':
                chan_labels = lumbar_chans
                colorbar_axes = [-0.1, 0.1]
            subjects_4grid = np.arange(1, 37)  # subj  # Pass this instead of (1, 37) for 1 subjects
            # you can also base the grid on an several subjects
            # then the function takes the average over the channel positions of all those subjects
            time = 0.0
            colorbar = True
            mrmr_esg_isopotentialplot(subjects_4grid, grand_average_spatial, colorbar_axes, chan_labels,
                                      colorbar, time, ax, colorbar_label='Amplitude (AU)')
            ax.set_yticklabels([])
            ax.set_ylabel(None)
            ax.set_xticklabels([])
            ax.set_xlabel(None)
            ax.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')

            ############################################################################################
            # Low Freq SEP
            ############################################################################################
            fig_low, ax_low = plt.subplots(1, 1)
            arrays = [np.array(x) for x in data_list]
            chanvalues = np.array([np.nanmean(k) for k in zip(*arrays)])

            # chan_labels = evoked.ch_names
            chan_labels = esg_chans
            if cond_name == 'median':
                colorbar_axes = [-0.3, 0.3]
            else:
                colorbar_axes = [-0.1, 0.1]
            subjects_4grid = np.arange(1, 37)
            # then the function takes the average over the channel positions of all those subjects
            colorbar = True
            mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                      time_point, ax_low, colorbar_label='Amplitude (\u03BCV)')
            ax_low.set_yticklabels([])
            ax_low.set_ylabel(None)
            ax_low.set_xticklabels([])
            ax_low.set_xlabel(None)
            ax_low.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')

            if use_visible is True:
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}_visible')
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}_visible.pdf',
                            bbox_inches='tight', format="pdf")

                fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible')
                fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible.pdf',
                                bbox_inches='tight', format="pdf")
            else:
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}')
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}.pdf',
                            bbox_inches='tight', format="pdf")

                fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}')
                fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}.pdf',
                                bbox_inches='tight', format="pdf")
