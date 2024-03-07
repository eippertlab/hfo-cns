# Plot grand average time courses and spatial patterns after application of CCA to EEG data
# Each time courses is shifted based on the latency of peak of the low-frequency response
# CAREFUL: For shifting it should be expected - sep_latency


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
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

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG.xlsx')
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
    df_timing = pd.read_excel(xls_timing, 'Cortical')
    df_timing.set_index('Subject', inplace=True)

    # Get a raw file so I can use the montage
    raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data/freq_banded_eeg/sub-001/sigma_median.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)
    figure_path = '/data/p_02718/Images/CCA_eeg/GrandAverage_Shifted/'
    os.makedirs(figure_path, exist_ok=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
    df_vis = pd.read_excel(xls, 'CCA_Brain')
    df_vis.set_index('Subject', inplace=True)

    use_visible = True  # Use only subjects with visible bursting

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
                evoked = epochs.copy().average()

                # Apply relative time-shift depending on expected latency
                if cond_name == 'median':
                    sep_latency = df_timing.loc[subject, f"N20"]
                    expected = 20 / 1000
                elif cond_name == 'tibial':
                    sep_latency = df_timing.loc[subject, f"P39"]
                    expected = 39 / 1000
                shift = expected - sep_latency
                evoked.shift_time(shift, relative=True)
                evoked.crop(tmin=-0.06, tmax=0.20)

                if use_visible is True:
                    visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                    if visible == 'T':
                        data = evoked.data
                        evoked_list.append(data)
                else:
                    data = evoked.data
                    evoked_list.append(data)

                ############################################################
                # Spatial Pattern Extraction
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
                else:
                    if inv == 'T':
                        spatial_pattern.append(A_st[:, channel_no - 1] * -1)
                    else:
                        spatial_pattern.append(A_st[:, channel_no - 1])

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
                ax.set_xlim([0.00, 0.05])
            else:
                ax.set_xlim([0.00, 0.07])

            if use_visible:
                plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}_visible')
                plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}_visible.pdf', bbox_inches='tight',
                            format="pdf")
            else:
                plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}')
                plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}.pdf', bbox_inches='tight', format="pdf")

            # Plot Spatial Pattern
            fig, ax = plt.subplots(1, 1)
            chan_labels = epochs.ch_names
            mne.viz.plot_topomap(data=grand_average_spatial*10**6, pos=res, ch_type='eeg', sensors=True, names=None,
                                 contours=6, outlines='head', sphere=None, image_interp='cubic',
                                 extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                 cnorm=None, axes=ax, show=False)
            ax.set_title(f'Grand Average Spatial Pattern, n={len(evoked_list)}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.6, orientation='vertical')
            cb.set_label('Amplitude', rotation=90)
            if use_visible:
                plt.savefig(figure_path + f'GA_Spatial_{freq_band}_{cond_name}_visible')
                plt.savefig(figure_path + f'GA_Spatial_{freq_band}_{cond_name}_visible.pdf', bbox_inches='tight',
                            format="pdf")
            else:
                plt.savefig(figure_path+f'GA_Spatial_{freq_band}_{cond_name}')
                plt.savefig(figure_path+f'GA_Spatial_{freq_band}_{cond_name}.pdf', bbox_inches='tight', format="pdf")
