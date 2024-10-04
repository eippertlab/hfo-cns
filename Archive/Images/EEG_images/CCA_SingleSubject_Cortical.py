# Plots each subjects chosen component after CCA has been performed on CNS_Level_Specific_Functions data
# Plots first 18 subjects in one subplot, and then the next 18 in another

import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
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

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    # Get a raw file so I can use the montage
    raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data/freq_banded_eeg/sub-001/sigma_median.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)
    figure_path = '/data/p_02718/Images/CCA_eeg/SingleSubject/'
    os.makedirs(figure_path, exist_ok=True)

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

                if cond_name == 'median':
                    sep_latency = 20  # In ms
                elif cond_name == 'tibial':
                    sep_latency = 40

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
                evoked = epochs.average()
                data = evoked.data

                # ############################################################
                # # Spatial Pattern Extraction
                # ############################################################
                # # Read in saved A_st
                # with open(f'{input_path}A_st_{freq_band}_{cond_name}.pkl', 'rb') as f:
                #     A_st = pickle.load(f)
                #     # Shape (channels, channel_rank)
                # if inv == 'T':
                #     spatial_pattern = (A_st[:, channel_no-1]*-1)
                # else:
                #     spatial_pattern = (A_st[:, channel_no-1])

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
                line_label = f"{sep_latency / 1000}s"
                plt.axvline(x=sep_latency / 1000, color='r', linewidth='0.6', label=line_label)
                if cond_name == 'median':
                    ax.set_xlim([0.00, 0.05])
                else:
                    ax.set_xlim([0.00, 0.07])

            fig1.tight_layout()
            fig2.tight_layout()
            fig1.savefig(figure_path + f'1_Time_subplots_{freq_band}_{cond_name}')
            fig2.savefig(figure_path + f'2_Time_subplots_{freq_band}_{cond_name}')
                # plt.savefig(figure_path+f'{subject_id}_Time_{freq_band}_{cond_name}')
                # plt.savefig(figure_path+f'{subject_id}_Time_{freq_band}_{cond_name}.pdf',
                #             bbox_inches='tight', format="pdf")

                # # Plot Spatial Pattern
                # fig, ax = plt.subplots(1, 1)
                # chan_labels = epochs.ch_names
                # mne.viz.plot_topomap(data=spatial_pattern*10**6, pos=res, ch_type='eeg', sensors=True, names=None,
                #                      contours=6, outlines='head', sphere=None, image_interp='cubic',
                #                      extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                #                      cnorm=None, axes=ax, show=False)
                # ax.set_title(f'Spatial Pattern, {subject_id}')
                # divider = make_axes_locatable(ax)
                # cax = divider.append_axes('right', size='5%', pad=0.05)
                # cb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.6, orientation='vertical')
                # cb.set_label('Amplitude', rotation=90)
                # plt.savefig(figure_path+f'{subject_id}_Spatial_{freq_band}_{cond_name}')
                # plt.savefig(figure_path+f'{subject_id}_Spatial_{freq_band}_{cond_name}.pdf',
                #             bbox_inches='tight', format="pdf")
