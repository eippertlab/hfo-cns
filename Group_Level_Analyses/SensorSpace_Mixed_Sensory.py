"""
Group level amplitude envelope for sensor space data
Aim is to compare mixed nerve stimulation (wrist) to sensory nerve stimulation (digit12)
Want to extract the amplitude in the CCA training window and compare between groups with a t-test
For sensory: Window shifted rightward by 4ms (upper limb) or 8ms (lower limb)
"""


import mne
import os
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_rel
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
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    freq_band = 'sigma'

    input_folders = {'Spinal': "/data/pt_02718/tmp_data_2/freq_banded_esg/",
                     'Cortical': "/data/pt_02718/tmp_data_2/freq_banded_eeg/"}

    # Timings
    tmin_cort = 15/1000
    tmax_cort = 25/1000
    shift = 4/1000 # Only for median now

    for level in ['Cortical']:
        amplitudes_m = []
        amplitudes_d = []
        evoked_list_mixed = []
        evoked_list_sensory = []
        srmr_nr = 2
        subjects = np.arange(1, 25)
        folder = 'tmp_data_2'
        figure_folder = 'Polished_2'

        figure_path = f'/data/p_02718/{figure_folder}/Raw_Envelopes_Mixed_Sensory/'
        os.makedirs(figure_path, exist_ok=True)

        fig, ax1 = plt.subplots(1, 2)
        ax1 = ax1.flatten()

        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path_sensory = f"{input_folders[level]}{subject_id}/sigma_med_digits.fif"
            input_path_mixed = f"{input_folders[level]}{subject_id}/sigma_med_mixed.fif"

            # med_mixed for wrist and med12 for sensory
            raw_dig = mne.io.read_raw_fif(input_path_sensory, preload=True)
            raw_mixed = mne.io.read_raw_fif(input_path_mixed, preload=True)

            # Average rereference
            if level == 'Cortical':
                raw_dig.set_eeg_reference(ref_channels='average')
                raw_mixed.set_eeg_reference(ref_channels='average')

            # Mixed data
            events, event_ids = mne.events_from_annotations(raw_mixed)
            event_id_dict = {key: value for key, value in event_ids.items() if key == 'medMixed'}
            epochs_m = mne.Epochs(raw_mixed, events, event_id=event_id_dict, tmin=iv_epoch[0],
                                  tmax=iv_epoch[1] - 1 / 1000,
                                  baseline=tuple(iv_baseline), preload=True)
            if level == 'Spinal':
                evoked_m = epochs_m.average().pick(['SC6'])
            else:
                evoked_m = epochs_m.average().pick(['CP4'])
            envelope_m = evoked_m.apply_hilbert(envelope=True)
            if level == 'Cortical':
                ch_name, latency, amplitude_m = envelope_m.get_peak(mode='pos', return_amplitude=True, tmin=tmin_cort,
                                                                    tmax=tmax_cort)
                amplitudes_m.append(amplitude_m)
            data_m = envelope_m.get_data()
            evoked_list_mixed.append(data_m)

            # Digits data
            events_d, event_ids_d = mne.events_from_annotations(raw_dig)
            event_id_dict_d = {key: value for key, value in event_ids_d.items() if key == 'med12'}
            epochs_d = mne.Epochs(raw_dig, events_d, event_id=event_id_dict_d, tmin=iv_epoch[0],
                                  tmax=iv_epoch[1] - 1 / 1000,
                                  baseline=tuple(iv_baseline), preload=True)
            if level == 'Spinal':
                evoked_d = epochs_d.average().pick(['SC6'])
            else:
                evoked_d = epochs_d.average().pick(['CP4'])
            envelope_d = evoked_d.apply_hilbert(envelope=True)
            if level == 'Cortical':
                ch_name, latency, amplitude_d = envelope_d.get_peak(mode='pos', return_amplitude=True, tmin=tmin_cort+shift,
                                                                    tmax=tmax_cort+shift)
                amplitudes_d.append(amplitude_d)
            data_d = envelope_d.get_data()
            evoked_list_sensory.append(data_d)

            # Plot single subject envelopes as you go
            ax1[0].plot(evoked_m.times, data_m.reshape(-1)*10**6)
            ax1[1].plot(evoked_d.times, data_d.reshape(-1)*10**6)

        ax1[0].set_title('Mixed')
        ax1[1].set_title('Sensory')
        ax1[0].set_xlabel('Time (s)')
        ax1[0].set_ylabel('Amplitude (\u03BCV)')
        ax1[0].set_xlim([0.0, 0.05])
        ax1[1].set_xlabel('Time (s)')
        ax1[1].set_ylabel('Amplitude (\u03BCV)')
        ax1[1].set_xlim([0.0, 0.05])
        plt.tight_layout()
        plt.savefig(figure_path + f'SingleSubject_Envelope_{level}')
        plt.savefig(figure_path + f'SingleSubject_Envelope_{level}.pdf',
                    bbox_inches='tight', format="pdf")
        plt.close(fig)

        #################################################################################################
        # Get grand average across chosen epochs and the standard error of the mean
        #################################################################################################
        fig, ax1 = plt.subplots()
        for label, evoked_list in zip(['Sensory', 'Mixed'],[evoked_list_sensory, evoked_list_mixed]):
            grand_average = np.mean(evoked_list, axis=0)
            error = sem(evoked_list, axis=0)
            upper = (grand_average[0, :] + error).reshape(-1)
            lower = (grand_average[0, :] - error).reshape(-1)

            #################################################################################################
            # Plot Time Course
            #################################################################################################
            ax1.plot(evoked_d.times, grand_average[0, :] * 10 ** 6, label=label)
            ax1.fill_between(evoked_d.times, lower * 10 ** 6, upper * 10 ** 6, alpha=0.3)

        ax1.set_xlabel('Time (s)')
        ax1.set_title(f'{level}')
        ax1.set_ylabel('Amplitude (\u03BCV)')
        ax1.set_xlim([0.0, 0.05])

        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path + f'GA_Envelope_{level}')
        plt.savefig(figure_path + f'GA_Envelope_{level}.pdf',
                    bbox_inches='tight', format="pdf")
        plt.close(fig)

        # Print the group average amplitude and standard deviation
        # Run a paired t-test between the groups
        if level == 'Cortical':
            print(f'Mixed Amplitude (uV): {np.mean(amplitudes_m)* 10 ** 6} + {sem(amplitudes_m, axis=0)* 10 ** 6}')
            print(f'Sensory Amplitude (uV): {np.mean(amplitudes_d)* 10 ** 6} + {sem(amplitudes_d, axis=0)* 10 ** 6}')
            result = ttest_rel(amplitudes_m, amplitudes_d)
            print(result)
