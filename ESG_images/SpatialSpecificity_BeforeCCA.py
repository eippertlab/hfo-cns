# Script to plot the time-frequency decomposition about the spinal triggers for the correct versus incorrect patch
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Common_Functions.evoked_from_raw import evoked_from_raw
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
    # iv_baseline = [-0.05, -0.01]
    # iv_epoch = [-0.06, 0.06]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path_singlesubject = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity/SingleSubject/"
    image_path_grandaverage = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity/GrandAverage/"
    os.makedirs(image_path_singlesubject, exist_ok=True)
    os.makedirs(image_path_grandaverage, exist_ok=True)

    subjects = np.arange(1, 37)
    sfreq = 5000
    cond_names = ['median', 'tibial']

    for freq_type in ['upper', 'full']:
        if freq_type == 'full':
            freqs = np.arange(0., 1000., 3.)
            fmin, fmax = freqs[[0, -1]]
        elif freq_type == 'upper':
            freqs = np.arange(200., 900., 3.)
            fmin, fmax = freqs[[0, -1]]

        # To use mne grand_average method, need to generate a list of evoked potentials for each subject
        for cond_name in cond_names:  # Conditions (median, tibial)
            evoked_list_correct = []
            evoked_list_incorrect = []

            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'Tibial - Stimulation'
                correct_channel = ['L1']
                incorrect_channel = ['SC6']

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'Median - Stimulation'
                correct_channel = ['SC6']
                incorrect_channel = ['L1']

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
                fname = f"ssp6_cleaned_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(esg_chans)

                evoked_correct = evoked.copy().pick_channels(correct_channel)
                evoked_incorrect = evoked.copy().pick_channels(incorrect_channel)

                power_correct = mne.time_frequency.tfr_stockwell(evoked_correct, fmin=fmin, fmax=fmax, width=3.0,
                                                                 n_jobs=5)
                power_incorrect = mne.time_frequency.tfr_stockwell(evoked_incorrect, fmin=fmin, fmax=fmax, width=3.0,
                                                                   n_jobs=5)
                evoked_list_correct.append(power_correct)
                evoked_list_incorrect.append(power_incorrect)

                # Generate Single Subject Images
                if cond_name == 'tibial':
                    tmin = 0.0
                    tmax = 0.035
                else:
                    tmin = 0.0
                    tmax = 0.025
                if freq_type == 'full':
                    vmin = 0
                    vmax = 80
                elif freq_type == 'upper':
                    vmin = 0
                    vmax = 15
                fig, ax = plt.subplots(1, 2)
                ax = ax.flatten()
                # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
                # power.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
                #            axes=ax, show=False, colorbar=True, dB=False,
                #            tmin=tmin, tmax=tmax, vmin=0)
                power_correct.plot([0], baseline=iv_baseline, mode='ratio', cmap='jet',
                                   axes=ax[0], show=False, colorbar=True, dB=False,
                                   tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                power_incorrect.plot([0], baseline=iv_baseline, mode='ratio', cmap='jet',
                                     axes=ax[1], show=False, colorbar=True, dB=False,
                                     tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax)
                im = ax[0].images
                cb = im[-1].colorbar
                cb.set_label('Amplitude')
                im = ax[1].images
                cb = im[-1].colorbar
                cb.set_label('Amplitude')
                ax[0].set_title(f"Subject {subject} TFR\n"
                                f"Correct Patch")
                ax[1].set_title(f"Subject {subject} TFR\n"
                                f"Incorrect Patch")
                if freq_type == 'full':
                    fname = f"{subject_id}_{trigger_name}_full_ratio"
                elif freq_type == 'upper':
                    fname = f"{subject_id}_{trigger_name}_ratio"
                plt.tight_layout()
                fig.savefig(image_path_singlesubject + fname+'.png')
                plt.savefig(image_path_singlesubject + fname+'.pdf', bbox_inches='tight', format="pdf")
                # exit()
                plt.clf()

            averaged_correct = mne.grand_average(evoked_list_correct, interpolate_bads=False, drop_bads=False)
            averaged_incorrect = mne.grand_average(evoked_list_incorrect, interpolate_bads=False, drop_bads=False)
            # relevant_channel = averaged.pick_channels(channel)

            # power = mne.time_frequency.tfr_stockwell(relevant_channel, fmin=fmin, fmax=fmax, width=1.0, n_jobs=5)
            fig, ax = plt.subplots(1, 2)
            ax = ax.flatten()
            # averaged.plot([0], baseline=iv_baseline, mode='mean', cmap='jet',
            #               axes=ax, show=False, colorbar=True, dB=False,
            #               tmin=tmin, tmax=tmax, vmin=0)
            vmax = 120
            averaged_correct.plot([0], baseline=iv_baseline, mode='ratio', cmap='jet',
                                  axes=ax[0], show=False, colorbar=True, dB=False,
                                  tmin=tmin, tmax=tmax, vmin=0, vmax=vmax)
            averaged_incorrect.plot([0], baseline=iv_baseline, mode='ratio', cmap='jet',
                                    axes=ax[1], show=False, colorbar=True, dB=False,
                                    tmin=tmin, tmax=tmax, vmin=0, vmax=vmax)
            im = ax[0].images
            cb = im[-1].colorbar
            cb.set_label('Amplitude')
            im = ax[1].images
            cb = im[-1].colorbar
            cb.set_label('Amplitude')
            ax[0].set_title(f"Grand Average TFR\n"
                            f"Correct Patch")
            ax[1].set_title(f"Grand Average TFR\n"
                            f"Incorrect Patch")
            if freq_type == 'full':
                fname = f"{trigger_name}_full_ratio"
            elif freq_type == 'upper':
                fname = f"{trigger_name}_ratio"
            plt.tight_layout()
            fig.savefig(image_path_grandaverage + fname+'.png')
            plt.savefig(image_path_grandaverage + fname+'.pdf', bbox_inches='tight', format="pdf")
            plt.clf()

