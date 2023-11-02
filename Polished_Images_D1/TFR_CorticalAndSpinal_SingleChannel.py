# Script to plot the time-frequency decomposition about the spinal triggers for the correct versus incorrect patch
# Also include the cortical data for just the correct cluster
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

    shift_spinal = False  # If true, shift the spinal based on time of underlying low freq SEP

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    image_path = "/data/p_02718/Polished/TFRs_SingleChannel/"
    os.makedirs(image_path, exist_ok=True)

    xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
    df_timing_spinal = pd.read_excel(xls_timing, 'Spinal')
    df_timing_spinal.set_index('Subject', inplace=True)

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
            evoked_list_cortical = []
            evoked_list_correct = []
            evoked_list_incorrect = []

            if cond_name == 'tibial':
                full_name = 'Tibial Nerve Stimulation'
                trigger_name = 'Tibial - Stimulation'
                correct_channel = ['L1']
                incorrect_channel = ['SC6']
                cortical_channel = ['Cz']

            elif cond_name == 'median':
                full_name = 'Median Nerve Stimulation'
                trigger_name = 'Median - Stimulation'
                correct_channel = ['SC6']
                incorrect_channel = ['L1']
                cortical_channel = ['CP4']

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Read in cortical data
                input_path = f"/data/pt_02718/tmp_data/imported/{subject_id}" + "/"
                raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr5000_{cond_name}_withqrs_eeg.fif", preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked_cortical = evoked.copy().pick_channels(cortical_channel)

                # Read in Spinal Data
                input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
                fname = f"ssp6_cleaned_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked_spinal = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked_correct = evoked_spinal.copy().pick_channels(correct_channel)
                evoked_incorrect = evoked_spinal.copy().pick_channels(incorrect_channel)

                if shift_spinal:
                    # Apply relative time-shift depending on expected latency for spinal data
                    # median_lat, tibial_lat = get_time_to_align('esg', ['median', 'tibial'], np.arange(1, 37))
                    median_lat = 0.013
                    tibial_lat = 0.022
                    for evoked in [evoked_correct, evoked_incorrect]:
                        if cond_name == 'median':
                            sep_latency = round(df_timing_spinal.loc[subject, f"N13"], 3)
                            expected = median_lat
                        elif cond_name == 'tibial':
                            sep_latency = round(df_timing_spinal.loc[subject, f"N22"], 3)
                            expected = tibial_lat
                        shift = expected - sep_latency
                        evoked.shift_time(shift, relative=True)
                        evoked.crop(tmin=-0.06, tmax=0.1)
                    else:
                        evoked_correct.crop(tmin=-0.06, tmax=0.1)
                        evoked_incorrect.crop(tmin=-0.06, tmax=0.1)

                    # No shift for cortical data
                    evoked_cortical.crop(tmin=-0.06, tmax=0.1)

                # Get power
                power_cortical = mne.time_frequency.tfr_stockwell(evoked_cortical, fmin=fmin, fmax=fmax, width=3.0,
                                                                  n_jobs=5)
                power_correct = mne.time_frequency.tfr_stockwell(evoked_correct, fmin=fmin, fmax=fmax, width=3.0,
                                                                 n_jobs=5)
                power_incorrect = mne.time_frequency.tfr_stockwell(evoked_incorrect, fmin=fmin, fmax=fmax, width=3.0,
                                                                   n_jobs=5)

                evoked_list_cortical.append(power_cortical)
                evoked_list_correct.append(power_correct)
                evoked_list_incorrect.append(power_incorrect)

            # Get grand average across subjects
            averaged_cortical = mne.grand_average(evoked_list_cortical, interpolate_bads=False, drop_bads=False)
            averaged_correct = mne.grand_average(evoked_list_correct, interpolate_bads=False, drop_bads=False)
            averaged_incorrect = mne.grand_average(evoked_list_incorrect, interpolate_bads=False, drop_bads=False)

            fig, ax = plt.subplots(1, 3, figsize=[18, 6])
            ax = ax.flatten()
            tmin = 0.0
            tmax = 0.06
            if cond_name == 'median':
                if freq_type == 'full':
                    vmin_cortical = 0
                    vmax_cortical = 30  # 160
                    vmin = 0
                    vmax = 15  # 80
                elif freq_type == 'upper':
                    vmin_cortical = 0
                    vmax_cortical = 30
                    vmin = 0
                    vmax = 15
            elif cond_name == 'tibial':
                if freq_type == 'full':
                    vmin_cortical = 0
                    vmax_cortical = 8  # 40
                    vmin = 0
                    vmax = 4  # 20
                elif freq_type == 'upper':
                    vmin_cortical = 0
                    vmax_cortical = 8
                    vmin = 0
                    vmax = 4
            # Because combine = 'mean', the data in all channels is averaged as picks = 'eeg'
            averaged_cortical.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                  axes=ax[0], show=False, colorbar=True, dB=False,
                                  tmin=tmin, tmax=tmax, vmin=0, vmax=vmax_cortical, combine='mean')
            averaged_correct.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                  axes=ax[1], show=False, colorbar=True, dB=False,
                                  tmin=tmin, tmax=tmax, vmin=0, vmax=vmax, combine='mean')
            averaged_incorrect.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                    axes=ax[2], show=False, colorbar=True, dB=False,
                                    tmin=tmin, tmax=tmax, vmin=0, vmax=vmax, combine='mean')
            for axes in [ax[0], ax[1], ax[2]]:
                im = axes.images
                cb = im[-1].colorbar
                cb.set_label('Amplitude (AU)')

            ax[0].set_title(f"Grand average cortical TFR\n"
                            f"Target electrode")
            if cond_name == 'median':
                ax[1].set_title(f"Grand average spinal TFR\n"
                                f"Target cervical electrode")
                ax[2].set_title(f"Grand average spinal TFR\n"
                                f"Non-target lumbar electrode")
            elif cond_name == 'tibial':
                ax[1].set_title(f"Grand average spinal TFR\n"
                                f"Target lumbar electrode")
                ax[2].set_title(f"Grand average spinal TFR\n"
                                f"Non-target cervical electrode")
            if freq_type == 'full':
                fname = f"{trigger_name}_full_ratio"
            elif freq_type == 'upper':
                fname = f"{trigger_name}_ratio"
            plt.tight_layout()
            if shift_spinal:
                fig.savefig(image_path + fname + '_spinalshifted_longcrop.png')
                plt.savefig(image_path + fname+'_spinalshifted_longcrop.pdf', bbox_inches='tight', format="pdf")
            else:
                fig.savefig(image_path + fname+'_longcrop.png')
                plt.savefig(image_path + fname+'_longcrop.pdf', bbox_inches='tight', format="pdf")
            plt.clf()
