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
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

if __name__ == '__main__':
    use_visible = True  # Use only subjects with visible bursting

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
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    figure_path = '/data/p_02718/Polished/GrandAverage_SpatialPatterns/'
    os.makedirs(figure_path, exist_ok=True)

    # Cortical Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
    df_vis_cortical = pd.read_excel(xls, 'CCA_Brain')
    df_vis_cortical.set_index('Subject', inplace=True)

    # Spinal Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
    df_vis_spinal = pd.read_excel(xls, 'CCA_Spinal')
    df_vis_spinal.set_index('Subject', inplace=True)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for data_type in ['esg', 'eeg']:
        for freq_band in freq_bands:
            for condition in conditions:
                spatial_pattern = []
                data_list = []
                evoked_list = []

                for subject in subjects:
                    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                    # Set variables
                    cond_info = get_conditioninfo(condition, srmr_nr)
                    cond_name = cond_info.cond_name
                    trigger_name = cond_info.trigger_name
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    # Select the right files
                    if data_type == 'eeg':
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                        df = df_cortical
                        df_vis = df_vis_cortical

                        # Low Freq SEP
                        input_path_low = "/data/pt_02068/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                        fname_low = f"cnt_clean_{cond_name}.set"
                        raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)

                        # Set montage
                        montage_path = '/data/pt_02718/'
                        montage_name = 'electrode_montage_eeg_10_5.elp'
                        montage = mne.channels.read_custom_montage(montage_path + montage_name)
                        raw.set_montage(montage, on_missing="ignore")
                        idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
                        res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

                        evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                        if trigger_name == 'Median - Stimulation':
                            time_point = 18.8 / 1000
                            cond_name = 'median'
                            channel = ['CP4']
                        else:
                            cond_name = 'tibial'
                            time_point = 40.4 / 1000
                            channel = ['Cz']

                    else:
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                        df = df_spinal
                        df_vis = df_vis_spinal

                        # Low Freq SEP
                        input_path_low = f"/data/p_02569/SSP_forhfo/{subject_id}/6 projections/"
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
                            if data_type == 'esg':
                                evoked_low.crop(tmin=time_point, tmax=time_point + (2 / 1000))
                                data = evoked_low.data.mean(axis=1)
                                data_list.append(data)
                            elif data_type == 'eeg':
                                evoked_list.append(evoked_low)
                    else:
                        if inv == 'T':
                            spatial_pattern.append(A_st[:, channel_no - 1] * -1)
                        else:
                            spatial_pattern.append(A_st[:, channel_no - 1])

                        if data_type == 'esg':
                            evoked_low.crop(tmin=time_point, tmax=time_point + (2 / 1000))
                            data = evoked_low.data.mean(axis=1)
                            data_list.append(data)
                        elif data_type == 'eeg':
                            evoked_list.append(evoked_low)

                # Get grand average
                grand_average_spatial = np.mean(spatial_pattern, axis=0)  # HFO

                if data_type == 'eeg':
                    averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)  # SEP for eeg
                    #################################################################################################
                    # HFO
                    #################################################################################################
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    chan_labels = evoked_low.ch_names
                    mne.viz.plot_topomap(data=grand_average_spatial * 10 ** 6, pos=res, ch_type='eeg', sensors=True,
                                         names=None,
                                         contours=6, outlines='head', sphere=None, image_interp='cubic',
                                         extrapolate='head', border='mean', res=64, size=1, cmap='RdBu_r', vlim=(None, None),
                                         cnorm=None, axes=ax, show=False)
                    # ax.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.6, orientation='vertical')
                    cb.set_label('Amplitude (AU)', rotation=90)

                    ###############################################################################################
                    # Low Freq SEP
                    ###############################################################################################
                    fig_low, ax_low = plt.subplots(1, 1, figsize=(10, 10))
                    # divider = make_axes_locatable(plt.gca())
                    # cax = divider.append_axes("right", "5%", pad="3%")

                    # fig_low = plt.figure()
                    # ax_low = plt.subplot2grid(shape=(10, 25), loc=(0, 0), colspan=24, rowspan=10)
                    # cax = plt.subplot2grid(shape=(10, 25), loc=(1, 24), colspan=1, rowspan=8)
                    averaged.plot_topomap(times=time_point, average=None, ch_type=None, scalings=None, proj=False,
                                          sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                          outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                          border='mean',
                                          res=64, size=1, cmap='RdBu_r',
                                          colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_low, time_unit='s',
                                          time_format=None,
                                          nrows=1, ncols='auto', show=True)
                    # ax_low.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')
                    divider = make_axes_locatable(ax_low)
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    cb = fig.colorbar(ax_low.images[-1], cax=cax, shrink=0.6, orientation='vertical')
                    cb.set_label('Amplitude (\u03BCV)', rotation=90)

                elif data_type == 'esg':
                    ##########################################################################################
                    # HFO
                    ##########################################################################################
                    fig, ax = plt.subplots(figsize=(10, 10))
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
                                              colorbar, time, ax, colorbar_label='Amplitude (AU)', srmr_nr=srmr_nr)
                    ax.set_yticklabels([])
                    ax.set_ylabel(None)
                    ax.set_xticklabels([])
                    ax.set_xlabel(None)
                    # ax.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')

                    ############################################################################################
                    # Low Freq SEP
                    ############################################################################################
                    fig_low, ax_low = plt.subplots(1, 1, figsize=(10, 10))
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
                                              time_point, ax_low, colorbar_label='Amplitude (\u03BCV)', srmr_nr=srmr_nr)
                    ax_low.set_yticklabels([])
                    ax_low.set_ylabel(None)
                    ax_low.set_xticklabels([])
                    ax_low.set_xlabel(None)
                    # ax_low.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')

                if use_visible is True:
                    fig.savefig(figure_path + f'{data_type}_HFO_GA_Spatial_{freq_band}_{cond_name}_visible')
                    fig.savefig(figure_path + f'{data_type}_HFO_GA_Spatial_{freq_band}_{cond_name}_visible.pdf',
                                bbox_inches='tight', format="pdf")

                    fig_low.savefig(figure_path + f'{data_type}_SEP_GA_Spatial_{cond_name}_visible')
                    fig_low.savefig(figure_path + f'{data_type}_SEP_GA_Spatial_{cond_name}_visible.pdf',
                                    bbox_inches='tight', format="pdf")
                else:
                    fig.savefig(figure_path + f'{data_type}_HFO_GA_Spatial_{freq_band}_{cond_name}')
                    fig.savefig(figure_path + f'{data_type}_HFO_GA_Spatial_{freq_band}_{cond_name}.pdf',
                                bbox_inches='tight', format="pdf")

                    fig_low.savefig(figure_path + f'{data_type}_SEP_GA_Spatial_{cond_name}')
                    fig_low.savefig(figure_path + f'{data_type}_SEP_GA_Spatial_{cond_name}.pdf',
                                    bbox_inches='tight', format="pdf")
