# Plot grand average spatial patterns after application of CCA for HFOs
# Plot grand average spatial patterns for low frequency raw potentials
# Doing this for THALAMIC activity - only for visible subjects


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
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
    for srmr_nr in [1, 2]:

        if srmr_nr == 1:
            subjects = np.arange(1, 37)
            conditions = [2, 3]
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

        if srmr_nr == 1:
            folder = 'tmp_data'
            figure_path = '/data/p_02718/Polished/GrandAverage_SpatialPatterns_Thalamic/'
            os.makedirs(figure_path, exist_ok=True)

        elif srmr_nr == 2:
            folder = 'tmp_data_2'
            figure_path = '/data/p_02718/Polished_2/GrandAverage_SpatialPatterns_Thalamic/'
            os.makedirs(figure_path, exist_ok=True)

        # Excel files
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Thalamic_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA')
        df_cortical.set_index('Subject', inplace=True)

        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Thalamic_Updated.xlsx')
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
                    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

                    # HFO
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = f"/data/pt_02718/{folder}/cca_eeg_thalamic/{subject_id}/"
                    df = df_cortical
                    df_vis = df_vis_cortical

                    # Low Freq SEP
                    input_path_low = f"/data/pt_02718/{folder}/imported/{subject_id}/"
                    fname_low = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"
                    raw = mne.io.read_raw_fif(input_path_low + fname_low, preload=True)

                    # Set montage
                    montage_path = '/data/pt_02718/'
                    montage_name = 'electrode_montage_eeg_10_5.elp'
                    montage = mne.channels.read_custom_montage(montage_path + montage_name)
                    raw.set_montage(montage, on_missing="ignore")
                    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
                    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

                    # evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    events, event_ids = mne.events_from_annotations(raw)
                    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                    epochs_low = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                                        baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)
                    evoked_low = epochs_low.average()
                    evoked_low.reorder_channels(eeg_chans)

                    if cond_name in ['median', 'med_mixed']:
                        time_point = 15 / 1000
                        channel = ['CP4']
                    else:
                        time_point = 30 / 1000
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
                    visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                    if visible == 'T':
                        if inv == 'T':
                            spatial_pattern.append(A_st[:, channel_no - 1] * -1)
                        else:
                            spatial_pattern.append(A_st[:, channel_no - 1])
                        evoked_list.append(evoked_low)

                # Get grand average
                grand_average_spatial = np.mean(spatial_pattern, axis=0)  # HFO
                averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)  # SEP for eeg

                #################################################################################################
                # HFO
                #################################################################################################
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                chan_labels = evoked_low.ch_names
                if srmr_nr == 1:
                    if cond_name == 'tibial':
                        vmin = 0.000
                        vmax = 0.120
                    elif cond_name == 'median':
                        vmin = 0.000
                        vmax = 0.40
                elif srmr_nr == 2:
                    if cond_name == 'tib_mixed':
                        vmin = 0.000
                        vmax = 0.20
                    elif cond_name == 'med_mixed':
                        vmin = 0.000
                        vmax = 0.50
                mne.viz.plot_topomap(data=grand_average_spatial * 10 ** 6, pos=res, ch_type='eeg', sensors=True,
                                     names=None,
                                     contours=6, outlines='head', sphere=None, image_interp='cubic',
                                     extrapolate='box', border='mean', res=64, size=1, cmap='RdBu_r', vlim=(vmin, vmax),
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
                if srmr_nr == 1:
                    if cond_name == 'tibial':
                        vmin = -0.3
                        vmax = 0.6
                    elif cond_name == 'median':
                        vmin = -1.2
                        vmax = 1.2
                elif srmr_nr == 2:
                    if cond_name == 'tib_mixed':
                        vmin = None
                        vmax = None
                    elif cond_name == 'med_mixed':
                        vmin = -1
                        vmax = 0.8
                averaged.plot_topomap(times=time_point, average=None, ch_type=None, scalings=None, proj=False,
                                      sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                      outlines='head', sphere=None, image_interp='cubic', extrapolate='box',
                                      border='mean',
                                      res=64, size=1, cmap='RdBu_r',
                                      colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_low, time_unit='s',
                                      time_format=None,
                                      nrows=1, ncols='auto', show=False)
                # ax_low.set_title(f'Grand Average Spatial Pattern, n={len(spatial_pattern)}')
                divider = make_axes_locatable(ax_low)
                ax_low.set_title('')
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = fig.colorbar(ax_low.images[-1], cax=cax, shrink=0.6, orientation='vertical')
                cb.set_label('Amplitude (\u03BCV)', rotation=90)

                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}_visible_box')
                fig.savefig(figure_path + f'HFO_GA_Spatial_{freq_band}_{cond_name}_visible_box.pdf',
                            bbox_inches='tight', format="pdf")

                fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_box')
                fig_low.savefig(figure_path + f'SEP_GA_Spatial_{cond_name}_visible_box.pdf',
                                bbox_inches='tight', format="pdf")