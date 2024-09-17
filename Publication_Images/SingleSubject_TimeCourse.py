# Plot single subject time courses after application of CCA


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import matplotlib as mpl
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
mpl.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':
    srmr_nr = 2
    freq_bands = ['sigma']

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
        figure_folder = 'Polished'
        subjects_cortical = [3, 6, 31, 36]
        subjects_subcortical = [3, 6, 31, 36]
        subjects_spinal = [3, 6, 31, 36]

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'
        figure_folder = 'Polished_2'
        subjects_cortical = [10, 14, 20]  # [2, 5, 7, 13, 17, 19, 22]
        subjects_subcortical = [10, 14, 20]  # [2, 5, 7, 13, 17, 19, 22]
        subjects_spinal = [10, 14, 20]  # [2, 5, 7, 13, 17, 19, 22]

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    figure_path = f'/data/p_02718/{figure_folder}/SingleSubject/'
    os.makedirs(figure_path, exist_ok=True)

    # Cortical Excel files
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Updated.xlsx')
    df_vis_cortical = pd.read_excel(xls, 'CCA_Brain')
    df_vis_cortical.set_index('Subject', inplace=True)

    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    # Subcortical Excel files
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Thalamic_Updated.xlsx')
    df_vis_sub = pd.read_excel(xls, 'CCA_Brain')
    df_vis_sub.set_index('Subject', inplace=True)

    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Thalamic_Updated.xlsx')
    df_sub = pd.read_excel(xls, 'CCA')
    df_sub.set_index('Subject', inplace=True)

    # Spinal Excel files
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Visibility_Updated.xlsx')
    df_vis_spinal = pd.read_excel(xls, 'CCA_Spinal')
    df_vis_spinal.set_index('Subject', inplace=True)

    for data_type, subjects in zip(['eeg', 'esg', 'eeg_sub'], [subjects_cortical, subjects_spinal, subjects_subcortical]):
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
                    if data_type == 'eeg':
                        color_env = 'tab:purple'
                        color = 'indigo'

                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = f"/data/pt_02718/{folder}/cca_eeg/" + subject_id + "/"
                        df = df_cortical
                        df_vis = df_vis_cortical

                    elif data_type == 'eeg_sub':
                        color_env = 'tab:cyan'
                        color = 'teal'
                        color_low = 'seagreen'

                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = f"/data/pt_02718/{folder}/cca_eeg_thalamic/{subject_id}/"
                        df = df_sub
                        df_vis = df_vis_sub

                    else:
                        color_env = 'tab:blue'
                        color = 'royalblue'
                        color_low = 'deeppink'
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = f"/data/pt_02718/{folder}/cca/" + subject_id + "/"
                        df = df_spinal
                        df_vis = df_vis_spinal

                    epochs = mne.read_epochs(input_path + fname, preload=True)

                    # Need to pick channel based on excel sheet
                    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    channel = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    data = evoked.data

                    # Get envelope to add to high frequency plot
                    evoked.crop(tmin=-0.06, tmax=0.07)
                    envelope = evoked.copy().apply_hilbert(envelope=True)
                    data_envelope = envelope.get_data()

                    # Plot Time Course as YY plot
                    fig, ax1 = plt.subplots(1, 1)

                    # HFO
                    ax1.plot(evoked.times, evoked.data.reshape(-1), color=color)
                    ax1.plot(evoked.times, data_envelope[0, :], color=color_env, alpha=0.7)
                    ax1.set_ylabel('HFO Amplitude (AU)')
                    ax1.set_xlabel('Time (s)')

                    # ax1.set_title(f'Subject {subject} Time Courses')
                    ax1.set_xlim([0.00, 0.07])
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)

                    if cond_name in ['median', 'med_mixed']:
                        if data_type == 'eeg':
                            ax1.axvline(0.02, color='black', label='20ms', linestyle='dashed')
                        elif data_type == 'eeg_sub':
                            ax1.axvline(0.014, color='black', label='14ms', linestyle='dashed')
                        elif data_type == 'esg':
                            ax1.axvline(0.013, color='black', label='13ms', linestyle='dashed')
                    elif cond_name in ['tibial', 'tib_mixed']:
                        if data_type == 'eeg':
                            ax1.axvline(0.04, color='black', label='40ms', linestyle='dashed')
                        elif data_type == 'eeg_sub':
                            ax1.axvline(0.03, color='black', label='30ms', linestyle='dashed')
                        elif data_type == 'esg':
                            ax1.axvline(0.022, color='black', label='22ms', linestyle='dashed')

                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(figure_path + f'{data_type}_{subject_id}_Time_{freq_band}_{cond_name}')
                    plt.savefig(figure_path + f'{data_type}_{subject_id}_Time_{freq_band}_{cond_name}.pdf',
                                bbox_inches='tight', format="pdf")