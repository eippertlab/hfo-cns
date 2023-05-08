# Script to plot the time-frequency decomposition about the spinal triggers for the correct versus incorrect patch
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets
# Shifted based on latency of underlying low frequency potential

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

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    alternative_cluster = True  # USe the laternal electrodes in the patch too

    if alternative_cluster:
        image_path_singlesubject = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity_AltCluster_Shifted/SingleSubject/"
        image_path_grandaverage = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity_AltCluster_Shifted/GrandAverage/"
        os.makedirs(image_path_singlesubject, exist_ok=True)
        os.makedirs(image_path_grandaverage, exist_ok=True)
    else:
        image_path_singlesubject = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity_Cluster_Shifted/SingleSubject/"
        image_path_grandaverage = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity_Cluster_Shifted/GrandAverage/"
        os.makedirs(image_path_singlesubject, exist_ok=True)
        os.makedirs(image_path_grandaverage, exist_ok=True)

    subjects = np.arange(1, 37)
    sfreq = 5000
    cond_names = ['median', 'tibial']

    plot_single_subject = True
    plot_grand_average = True

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
            evoked_list_difference = []

            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'Tibial - Stimulation'
                if alternative_cluster:
                    correct_channel = ['S23', 'L1', 'S31', 'S26', 'S30', 'S28', 'S32']
                    incorrect_channel = ['S6', 'SC6', 'S14', 'S9', 'S13', 'S11', 'S15']
                else:
                    correct_channel = ['S23', 'L1', 'S31']
                    incorrect_channel = ['S6', 'SC6', 'S14']

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'Median - Stimulation'
                if alternative_cluster:
                    correct_channel = ['S6', 'SC6', 'S14', 'S9', 'S13', 'S11', 'S15']
                    incorrect_channel = ['S23', 'L1', 'S31', 'S26', 'S30', 'S28', 'S32']
                else:
                    correct_channel = ['S6', 'SC6', 'S14']
                    incorrect_channel = ['S23', 'L1', 'S31']

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
                fname = f"ssp6_cleaned_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked.reorder_channels(esg_chans)

                evoked_correct = evoked.copy().pick_channels(correct_channel)
                evoked_incorrect = evoked.copy().pick_channels(incorrect_channel)

                # Apply relative time-shift depending on expected latency
                potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
                fname_pot = 'potential_latency.mat'
                matdata = loadmat(potential_path + fname_pot)
                for evoked in [evoked_correct, evoked_incorrect]:
                    if cond_name == 'median':
                        sep_latency = matdata['med_potlatency']
                        expected = 13 / 1000
                    elif cond_name == 'tibial':
                        sep_latency = matdata['tib_potlatency']
                        expected = 22 / 1000
                    shift = sep_latency[0][0] / 1000 - expected
                    evoked.shift_time(shift, relative=True)
                    evoked.crop(tmin=-0.06, tmax=0.06)

                power_correct = mne.time_frequency.tfr_stockwell(evoked_correct, fmin=fmin, fmax=fmax, width=3.0,
                                                                 n_jobs=5)
                power_incorrect = mne.time_frequency.tfr_stockwell(evoked_incorrect, fmin=fmin, fmax=fmax, width=3.0,
                                                                   n_jobs=5)
                # Get the correct minus the incorrect channel
                power_difference = power_correct - power_incorrect
                evoked_list_difference.append(power_difference)

                evoked_list_correct.append(power_correct)
                evoked_list_incorrect.append(power_incorrect)

                if cond_name == 'tibial':
                    tmin = 0.0
                    tmax = 0.035
                else:
                    tmin = 0.0
                    tmax = 0.025
                if plot_single_subject:
                    # Generate Single Subject Images
                    if cond_name == 'median':
                        if freq_type == 'full':
                            vmin = 0
                            vmax = 80
                        elif freq_type == 'upper':
                            vmin = 0
                            vmax = 15
                    elif cond_name == 'tibial':
                        if freq_type == 'full':
                            vmin = 0
                            vmax = 20
                        elif freq_type == 'upper':
                            vmin = 0
                            vmax = 4
                    fig, ax = plt.subplots(1, 2)
                    ax = ax.flatten()
                    power_correct.plot('eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                       axes=ax[0], show=False, colorbar=True, dB=False,
                                       tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax, combine='mean')
                    power_incorrect.plot('eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                         axes=ax[1], show=False, colorbar=True, dB=False,
                                         tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax, combine='mean')

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
            averaged_difference = mne.grand_average(evoked_list_difference, interpolate_bads=False, drop_bads=False)

            if plot_grand_average:
                fig, ax = plt.subplots(1, 2)
                ax = ax.flatten()
                if cond_name == 'median':
                    if freq_type == 'full':
                        vmin = 0
                        vmax = 80
                    elif freq_type == 'upper':
                        vmin = 0
                        vmax = 15
                elif cond_name == 'tibial':
                    if freq_type == 'full':
                        vmin = 0
                        vmax = 20
                    elif freq_type == 'upper':
                        vmin = 0
                        vmax = 4
                averaged_correct.plot('eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                      axes=ax[0], show=False, colorbar=True, dB=False,
                                      tmin=tmin, tmax=tmax, vmin=0, vmax=vmax, combine='mean')
                averaged_incorrect.plot('eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                        axes=ax[1], show=False, colorbar=True, dB=False,
                                        tmin=tmin, tmax=tmax, vmin=0, vmax=vmax, combine='mean')
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

                # Plot difference
                fig, ax = plt.subplots(1, 1)
                if cond_name == 'median':
                    if freq_type == 'full':
                        vmin = 0
                        vmax = 160
                    elif freq_type == 'upper':
                        vmin = 0
                        vmax = 30
                elif cond_name == 'tibial':
                    if freq_type == 'full':
                        vmin = 0
                        vmax = 40
                    elif freq_type == 'upper':
                        vmin = 0
                        vmax = 15
                averaged_difference.plot('eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                         axes=ax, show=False, colorbar=True, dB=False,
                                         tmin=tmin, tmax=tmax, vmin=0, vmax=vmax, combine='mean')
                im = ax.images
                cb = im[-1].colorbar
                cb.set_label('Amplitude')
                ax.set_title(f"Grand Average TFR\n"
                             f"Correct - Incorrect Patch")
                if freq_type == 'full':
                    fname = f"{trigger_name}_full_ratio_difference"
                elif freq_type == 'upper':
                    fname = f"{trigger_name}_ratio_difference"
                plt.tight_layout()
                fig.savefig(image_path_grandaverage + fname + '.png')
                plt.savefig(image_path_grandaverage + fname + '.pdf', bbox_inches='tight', format="pdf")
                plt.clf()

