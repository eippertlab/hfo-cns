# Script to plot the time-frequency decomposition in dB about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Common_Functions.evoked_from_raw import evoked_from_raw
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    freqs = np.arange(400., 1800., 3.)
    fmin, fmax = freqs[[0, -1]]
    subjects = np.arange(1, 37)
    cond_names = ['median', 'tibial']

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]
    # iv_baseline = [-0.05, -0.01]
    # iv_epoch = [-0.06, 0.06]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path_singlesubject = "/data/p_02718/Images/TimeFrequencyPlots_Initial/SingleSubject/"
    image_path_grandaverage = "/data/p_02718/Images/TimeFrequencyPlots_Initial/GrandAverage/"
    os.makedirs(image_path_singlesubject, exist_ok=True)
    os.makedirs(image_path_grandaverage, exist_ok=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list = []

        if cond_name == 'tibial':
            full_name = 'Tibial Nerve Stimulation'
            trigger_name = 'Tibial - Stimulation'
            channel = ['L1']

        elif cond_name == 'median':
            full_name = 'Median Nerve Stimulation'
            trigger_name = 'Median - Stimulation'
            channel = ['SC6']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
            fname = f"ssp6_cleaned_{cond_name}.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked.reorder_channels(esg_chans)
            evoked = evoked.pick_channels(channel)
            power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)
            evoked_list.append(power)

            # Generate Single Subject Images
            if cond_name == 'tibial':
                tmin = 0.0
                tmax = 0.035
            else:
                tmin = 0.0
                tmax = 0.025
            # vmin = -200
            # vmax = -130
            fig, ax = plt.subplots(1, 1)
            # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            power.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                       axes=ax, show=False, colorbar=True, dB=False,
                       tmin=tmin, tmax=tmax, vmin=0)
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label('Amplitude [dB]')
            plt.title(f"Subject {subject} TFR\n"
                      f"Condition: {trigger_name}")
            fname = f"{subject_id}_{trigger_name}_dB.png"
            fig.savefig(image_path_singlesubject + fname)
            # exit()
            plt.clf()

        averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
        # relevant_channel = averaged.pick_channels(channel)

        # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
        fig, ax = plt.subplots(1, 1)
        averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                      axes=ax, show=False, colorbar=True, dB=False,
                      tmin=tmin, tmax=tmax, vmin=0)
        im = ax.images
        cb = im[-1].colorbar
        cb.set_label('Amplitude [dB]')
        plt.title(f"Grand Average TFR\n"
                  f"Condition: {trigger_name}")
        fname = f"{trigger_name}_dB.png"
        fig.savefig(image_path_grandaverage+fname)
        plt.clf()

