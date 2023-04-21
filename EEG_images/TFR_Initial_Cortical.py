# Script to plot the time-frequency decomposition in dB about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    image_path_singlesubject = "/data/p_02718/Images/TimeFrequencyPlots_InitialCortical/SingleSubject/"
    image_path_grandaverage = "/data/p_02718/Images/TimeFrequencyPlots_InitialCortical/GrandAverage/"
    os.makedirs(image_path_singlesubject, exist_ok=True)
    os.makedirs(image_path_grandaverage, exist_ok=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, 1)

    subjects = np.arange(1, 37)
    sfreq = 5000
    cond_names = ['median', 'tibial']

    for freq_type in ['200', '300']:
        if freq_type == 'full':
            freqs = np.arange(0., 1800., 3.)
            fmin, fmax = freqs[[0, -1]]
        elif freq_type == 'upper':
            freqs = np.arange(400., 1800., 3.)
            fmin, fmax = freqs[[0, -1]]
        elif freq_type == '200':
            freqs = np.arange(200., 1800., 3.)
            fmin, fmax = freqs[[0, -1]]
        elif freq_type == '300':
            freqs = np.arange(300., 1800., 3.)
            fmin, fmax = freqs[[0, -1]]

        # To use mne grand_average method, need to generate a list of evoked potentials for each subject
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list = []

            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'Tibial - Stimulation'
                channel = ['Cz']

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'Median - Stimulation'
                channel = ['CP4']

            for subject in subjects:  # All subjects
                bad_flag = False
                subject_id = f'sub-{str(subject).zfill(3)}'
                input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
                fname = f"noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(eeg_chans)
                evoked = evoked.pick_channels(channel)
                if channel[0] in evoked.info['bads']:
                    evoked.info['bads'] = []
                    bad_flag = True
                power = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)
                evoked_list.append(power)

                # Generate Single Subject Images
                if cond_name == 'tibial':
                    tmin = 0.0
                    tmax = 0.07
                else:
                    tmin = 0.0
                    tmax = 0.05
                # vmin = -200
                # vmax = -130
                fig, ax = plt.subplots(1, 1)
                power.plot([0], baseline=iv_baseline, mode='ratio', cmap='jet',
                           axes=ax, show=False, colorbar=True, dB=False,
                           tmin=tmin, tmax=tmax, vmin=0)
                im = ax.images
                cb = im[-1].colorbar
                cb.set_label('Amplitude')
                if bad_flag is True:
                    plt.title(f"Subject {subject} TFR\n"
                              f"Condition: {trigger_name}, Bad Channel")
                else:
                    plt.title(f"Subject {subject} TFR\n"
                              f"Condition: {trigger_name}")
                if freq_type == 'full':
                    fname = f"{subject_id}_{trigger_name}_full_ratio"
                elif freq_type == 'upper':
                    fname = f"{subject_id}_{trigger_name}_ratio"
                elif freq_type == '200':
                    fname = f"{subject_id}_{trigger_name}_ratio_200"
                elif freq_type == '300':
                    fname = f"{subject_id}_{trigger_name}_ratio_300"
                fig.savefig(image_path_singlesubject + fname + '.png')
                plt.savefig(image_path_singlesubject + fname + '.pdf', bbox_inches='tight', format="pdf")
                # exit()
                plt.clf()

            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
            # relevant_channel = averaged.pick_channels(channel)

            # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            fig, ax = plt.subplots(1, 1)
            # averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
            #               axes=ax, show=False, colorbar=True, dB=False,
            #               tmin=tmin, tmax=tmax, vmin=0)
            averaged.plot([0], baseline=iv_baseline, mode='ratio', cmap='jet',
                          axes=ax, show=False, colorbar=True, dB=False,
                          tmin=tmin, tmax=tmax, vmin=0)
            # , vmin=0, vmax=0.7e-15
            im = ax.images
            cb = im[-1].colorbar
            cb.set_label('Amplitude')
            plt.title(f"Grand Average TFR\n"
                      f"Condition: {trigger_name}")
            if freq_type == 'full':
                fname = f"{trigger_name}_full_ratio"
            elif freq_type == 'upper':
                fname = f"{trigger_name}_ratio"
            elif freq_type == '200':
                fname = f"{trigger_name}_ratio_200"
            elif freq_type == '300':
                fname = f"{trigger_name}_ratio_300"
            fig.savefig(image_path_grandaverage+fname+'.png')
            plt.savefig(image_path_grandaverage+fname+'.pdf', bbox_inches='tight', format="pdf")
            plt.clf()
