# Looking at the average power in a region of interest (400-800Hz), comparing this between the 'correct' and 'incorrect'
# patch in the spinal cord after median/tibial nerve stimulation

# Script to plot the time-frequency decomposition about the spinal triggers for the correct versus incorrect patch
# Also include the cortical data for just the correct cluster
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from scipy.io import loadmat
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.check_excel_exist_power import check_excel_exist_power
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 2
    freq_band = 'sigma'
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    if srmr_nr == 1:
        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
        df_timing_spinal = pd.read_excel(xls_timing, 'Spinal')
        df_timing_spinal.set_index('Subject', inplace=True)

        figure_path = f'/data/p_02718/Images/SpatialSpecificity_AmplitudeEnvelopes/'
        os.makedirs(figure_path, exist_ok=True)

        subjects = np.arange(1, 37)
        sfreq = 5000
        conditions = [2, 3]

    elif srmr_nr == 2:
        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx')
        df_timing_spinal = pd.read_excel(xls_timing, 'Spinal')
        df_timing_spinal.set_index('Subject', inplace=True)

        figure_path = f'/data/p_02718/Images_2/SpatialSpecificity_AmplitudeEnvelopes/'
        os.makedirs(figure_path, exist_ok=True)

        subjects = np.arange(1, 25)
        sfreq = 5000
        conditions = [3, 5]

    for condition in conditions:
        ga_envelope_correct = []
        ga_envelope_incorrect = []
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            if cond_name in ['tibial', 'tib_mixed']:
                correct_channel = ['L1']
                incorrect_channel = ['SC6']
                time_peak = 0.022
                time_edge = 0.006

            elif cond_name in ['median', 'med_mixed']:
                correct_channel = ['SC6']
                incorrect_channel = ['L1']
                time_peak = 0.013
                time_edge = 0.003

            # Read in Spinal Data
            if srmr_nr == 1:
                input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
            elif srmr_nr == 2:
                input_path = "/data/pt_02718/tmp_data_2/freq_banded_esg/" + subject_id + "/"
            fname = f"{freq_band}_{cond_name}.fif"
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)
            evoked_spinal = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked_correct = evoked_spinal.copy().pick(correct_channel)
            evoked_incorrect = evoked_spinal.copy().pick(incorrect_channel)

            for evoked, ga in zip([evoked_correct, evoked_incorrect], [ga_envelope_correct, ga_envelope_incorrect]):
                evoked.crop(tmin=-0.06, tmax=0.07)
                envelope = evoked.apply_hilbert(envelope=True)
                data = envelope.get_data()
                ga.append(data)

        # Get grand average across chosen epochs
        average_correct = np.mean(ga_envelope_correct, axis=0)
        average_incorrect = np.mean(ga_envelope_incorrect, axis=0)

            # for channel, ga in zip([correct_channel, incorrect_channel], [ga_envelope_correct, ga_envelope_incorrect]):
            #     if cond_name in ['median', 'med_mixed']:
            #         evoked = evoked_spinal.copy().pick_channels(channel).crop(tmin=-0.1, tmax=0.05)
            #     elif cond_name in ['tibial', 'tib_mixed']:
            #         evoked = evoked_spinal.copy().pick_channels(channel).crop(tmin=-0.1, tmax=0.07)
            #
            #     envelope = evoked.copy().apply_hilbert(envelope=True)
            #     ga.append(envelope)

        # average_correct = mne.grand_average(ga_envelope_correct)
        # average_incorrect = mne.grand_average(ga_envelope_incorrect)
        fig, ax = plt.subplots(1, 1)
        ax.plot(evoked.times, average_correct[0, :], label='Correct Electrode')
        ax.plot(evoked.times, average_incorrect[0, :], label='Incorrect Electrode')
        if cond_name in ['median', 'med_mixed']:
            ax.set_xlim([0.0, 0.05])
        elif cond_name in ['tibial', 'tib_mixed']:
            ax.set_xlim([0.0, 0.07])
        plt.legend()
        plt.suptitle(f'EnvelopeComparison_ {cond_name}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        plt.tight_layout()
        plt.savefig(figure_path + f'EnvelopeComparison_{cond_name}')

