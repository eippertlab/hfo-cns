# Plot grand average spatial patterns after application of CCA for HFOs
# Plot grand average spatial patterns for low frequency raw potentials
# Doing this for THALAMIC activity


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
    reref_flag = False  # If true, rereference to FPz
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [3, 2]
        freq_bands = ['sigma']
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        freq_bands = ['sigma']

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    # Get a raw file so I can use the montage
    raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data/freq_banded_eeg/sub-001/sigma_median.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

    if srmr_nr == 1:
        figure_path = '/data/p_02718/Images/CCA_eeg_thalamic/GrandAverage_SpatialPatterns/'
        os.makedirs(figure_path, exist_ok=True)

        figure_path_ss = '/data/p_02718/Images/CCA_eeg_thalamic/SingleSubject_SpatialPatterns/'
        os.makedirs(figure_path_ss, exist_ok=True)

        # Excel files
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Thalamic_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA')
        df_cortical.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Thalamic_Updated.xlsx')
        df_vis_cortical = pd.read_excel(xls, 'CCA_Brain')
        df_vis_cortical.set_index('Subject', inplace=True)

    elif srmr_nr == 2:
        figure_path = '/data/p_02718/Images_2/CCA_eeg_thalamic/GrandAverage_SpatialPatterns/'
        os.makedirs(figure_path, exist_ok=True)

        figure_path_ss = '/data/p_02718/Images_2/CCA_eeg_thalamic/SingleSubject_SpatialPatterns/'
        os.makedirs(figure_path_ss, exist_ok=True)

        # Excel files
        xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA')
        df_cortical.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated.xlsx')
        df_vis_cortical = pd.read_excel(xls, 'CCA_Brain')
        df_vis_cortical.set_index('Subject', inplace=True)

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

                # HFO
                fname = f"{freq_band}_{cond_name}.fif"
                if srmr_nr == 1:
                    input_path = "/data/pt_02718/tmp_data/cca_eeg_thalamic/" + subject_id + "/"
                elif srmr_nr == 2:
                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg_thalamic/" + subject_id + "/"
                df = df_cortical
                df_vis = df_vis_cortical

                # Low Freq SEP
                if srmr_nr == 1:
                    input_path_low = "/data/pt_02068/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                elif srmr_nr == 2:
                    input_path_low = "/data/pt_02151/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                fname_low = f"cnt_clean_{cond_name}.set"
                raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)
                raw.set_montage(montage, on_missing="ignore")
                evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                if reref_flag:
                    if srmr_nr == 2:
                        evoked_low = evoked_low.add_rereference_channels('FPz')
                    evoked_low = evoked_low.set_eeg_reference(['FPz'])

                if cond_name in ['median', 'med_mixed']:
                    time_points = [13/1000, 14/1000, 15 / 1000, 16/1000, 17/1000]
                    time_point = 15/100
                    channel = ['CP4']
                else:
                    time_points = [28/1000, 29/1000, 30 / 1000, 31/1000, 32/1000]
                    time_point = 30/1000
                    channel = ['Cz']

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
                            current_pattern = A_st[:, channel_no - 1] * -1
                        else:
                            spatial_pattern.append(A_st[:, channel_no - 1])
                            current_pattern = A_st[:, channel_no - 1]
                        evoked_list.append(evoked_low)
                else:
                    if inv == 'T':
                        spatial_pattern.append(A_st[:, channel_no - 1] * -1)
                        current_pattern = A_st[:, channel_no - 1] * -1
                    else:
                        spatial_pattern.append(A_st[:, channel_no - 1])
                        current_pattern = A_st[:, channel_no - 1]

                    evoked_list.append(evoked_low)

                fig_low, ax_low = plt.subplots(1, len(time_points))
                evoked_low.plot_topomap(times=time_points, average=None, ch_type=None, scalings=None, proj=False,
                                    sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                    outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                    border='mean',
                                    res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None,
                                    cnorm=None,
                                    colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_low, time_unit='s',
                                    time_format=None,
                                    nrows=1, ncols='auto', show=False)
                # plt.title(f'{subject_id}, Spatial Pattern, {cond_name}')
                # divider = make_axes_locatable(ax_low)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # cb = fig.colorbar(ax_low.images[-1], cax=cax, shrink=0.6, orientation='vertical')
                # cb.set_label('Amplitude (\u03BCV)', rotation=90)

                if use_visible is True:
                    if reref_flag:
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}_visible_FPz')
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}_visible_FPz.pdf',
                                        bbox_inches='tight', format="pdf")
                    else:
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}_visible')
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}_visible.pdf',
                                        bbox_inches='tight', format="pdf")
                else:
                    if reref_flag:
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}_FPz')
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}_FPz.pdf',
                                        bbox_inches='tight', format="pdf")
                    else:
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}')
                        fig_low.savefig(figure_path_ss + f'{subject_id}_LowSpatial_{cond_name}.pdf',
                                        bbox_inches='tight', format="pdf")
                plt.close()

                #################################################################################################
                # HFO Single Subject
                #################################################################################################
                fig, ax = plt.subplots(1, 1)
                chan_labels = evoked_low.ch_names
                mne.viz.plot_topomap(data=current_pattern * 10 ** 6, pos=res, ch_type='eeg', sensors=True,
                                     names=None,
                                     contours=6, outlines='head', sphere=None, image_interp='cubic',
                                     extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                     cnorm=None, axes=ax, show=False)
                ax.set_title(f'{subject_id} Spatial Pattern, {cond_name}')
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # cb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.6, orientation='vertical')
                # cb.set_label('Amplitude (AU)', rotation=90)

                if use_visible is True:
                    fig.savefig(figure_path_ss + f'{subject_id}_HighSpatial_{cond_name}_visible')
                    fig.savefig(figure_path_ss + f'{subject_id}_HighSpatial_{cond_name}_visible.pdf',
                                bbox_inches='tight', format="pdf")
                else:
                    fig.savefig(figure_path_ss + f'{subject_id}_HighSpatial_{cond_name}')
                    fig.savefig(figure_path_ss + f'{subject_id}_HighSpatial_{cond_name}.pdf',
                                bbox_inches='tight', format="pdf")

            ##################################################################################################
            # GRAND AVERAGE
            ##################################################################################################
            grand_average_spatial = np.mean(spatial_pattern, axis=0)  # HFO
            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)  # SEP for eeg

            #################################################################################################
            # HFO grand average plot
            #################################################################################################
            fig, ax = plt.subplots(1, 1)
            chan_labels = evoked_low.ch_names
            mne.viz.plot_topomap(data=grand_average_spatial * 10 ** 6, pos=res, ch_type='eeg', sensors=True,
                                 names=None,
                                 contours=6, outlines='head', sphere=None, image_interp='cubic',
                                 extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                 cnorm=None, axes=ax, show=False)
            ax.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.6, orientation='vertical')
            cb.set_label('Amplitude (AU)', rotation=90)

            ###############################################################################################
            # Low Freq SEP average plot
            ###############################################################################################
            fig_low, ax_low = plt.subplots(1, 1)
            averaged.plot_topomap(times=time_point, average=None, ch_type=None, scalings=None, proj=False,
                                  sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                  outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                  border='mean',
                                  res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None,
                                  cnorm=None,
                                  colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_low, time_unit='s',
                                  time_format=None,
                                  nrows=1, ncols='auto', show=False)
            ax_low.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')
            divider = make_axes_locatable(ax_low)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(ax_low.images[-1], cax=cax, shrink=0.6, orientation='vertical')
            cb.set_label('Amplitude (\u03BCV)', rotation=90)

            if use_visible is True:
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}_visible')
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}_visible.pdf',
                            bbox_inches='tight', format="pdf")

                if reref_flag:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_FPz')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_FPz.pdf',
                                    bbox_inches='tight', format="pdf")
                else:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible.pdf',
                                    bbox_inches='tight', format="pdf")
            else:
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}')
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}.pdf',
                            bbox_inches='tight', format="pdf")

                if reref_flag:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_FPz')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_FPz.pdf',
                                    bbox_inches='tight', format="pdf")
                else:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}.pdf',
                                    bbox_inches='tight', format="pdf")

            ###############################################################################################
            # Low Freq SEP average plot - more time points
            ###############################################################################################
            if cond_name in ['median', 'med_mixed']:
                time_points = [13 / 1000, 14 / 1000, 15 / 1000, 16 / 1000, 17 / 1000]
                time_point = 15 / 100
                channel = ['CP4']
            else:
                time_points = [23/1000, 24/1000, 25/1000, 26/1000, 27/1000, 28 / 1000, 29 / 1000, 30 / 1000]
                time_point = 30 / 1000
                channel = ['Cz']
            fig_low, ax_low = plt.subplots(1, len(time_points))
            averaged.plot_topomap(times=time_points, average=None, ch_type=None, scalings=None, proj=False,
                                  sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                  outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                  border='mean',
                                  res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None,
                                  cnorm=None,
                                  colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_low, time_unit='s',
                                  time_format=None,
                                  nrows=1, ncols='auto', show=False)
            # ax_low.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')
            # divider = make_axes_locatable(ax_low)
            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # cb = fig.colorbar(ax_low.images[-1], cax=cax, shrink=0.6, orientation='vertical')
            # cb.set_label('Amplitude (\u03BCV)', rotation=90)

            if use_visible is True:
                if reref_flag:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_FPz_more')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_FPz_more.pdf',
                                    bbox_inches='tight', format="pdf")
                else:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_more')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_more.pdf',
                                    bbox_inches='tight', format="pdf")
            else:
                if reref_flag:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_FPz_more')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_FPz_more.pdf',
                                    bbox_inches='tight', format="pdf")
                else:
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_more')
                    fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_more.pdf',
                                    bbox_inches='tight', format="pdf")