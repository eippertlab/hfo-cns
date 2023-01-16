# Plot single subject envelopes


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_bands = ['sigma', 'kappa']
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

    figure_path = '/data/p_02718/Images/CCA_eeg/Envelope_SingleSubject/'
    os.makedirs(figure_path, exist_ok=True)

    single_subject = False  # If we want to do subject by subject analysis of envelope
    subject_subplots = True  # If we just want to plot them all and save

    if subject_subplots:
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
                    if cond_name == 'median':
                        evoked = epochs.crop(tmin=0.0, tmax=0.05).copy().average()
                    elif cond_name == 'tibial':
                        evoked = epochs.crop(tmin=0.0, tmax=0.07).copy().average()
                    envelope = evoked.apply_hilbert(envelope=True)
                    data = envelope.get_data()

                    # Plot Envelope
                    if subject <= 18:
                        ax = ax1[count1]
                        count1 += 1
                    else:
                        ax = ax2[count2]
                        count2 += 1
                    ax.plot(epochs.times, data.reshape(-1))
                    ax.set_xlabel('Time (s)')
                    ax.set_title(f'{subject_id}')
                    ax.set_ylabel('Amplitude')
                    if cond_name == 'median':
                        ax.set_xlim([0.0, 0.05])
                    else:
                        ax.set_xlim([0.0, 0.07])

                fig1.tight_layout()
                fig2.tight_layout()
                fig1.savefig(figure_path + f'1_Envelope_subplots_{freq_band}_{cond_name}')
                fig2.savefig(figure_path + f'2_Envelope_subplots_{freq_band}_{cond_name}')

    if single_subject:
        for freq_band in freq_bands:
            for condition in conditions:

                for subject in subjects:
                    # Set variables
                    cond_info = get_conditioninfo(condition, srmr_nr)
                    cond_name = cond_info.cond_name
                    trigger_name = cond_info.trigger_name
                    subject_id = f'sub-{str(subject).zfill(3)}'

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
                    if cond_name == 'median':
                        evoked = epochs.crop(tmin=0.0, tmax=0.05).copy().average()
                    elif cond_name == 'tibial':
                        evoked = epochs.crop(tmin=0.0, tmax=0.07).copy().average()
                    envelope = evoked.apply_hilbert(envelope=True)
                    data = envelope.get_data()

                    # Plot Envelope
                    fig, ax = plt.subplots()
                    ax.plot(epochs.times, data.reshape(-1))
                    ax.set_xlabel('Time (s)')
                    ax.set_title(f'Envelope, {subject_id }')
                    ax.set_ylabel('Amplitude')
                    if cond_name == 'median':
                        ax.set_xlim([0.0, 0.05])
                    else:
                        ax.set_xlim([0.0, 0.07])

                    plt.savefig(figure_path+f'{subject_id}_Envelope_{freq_band}_{cond_name}')
