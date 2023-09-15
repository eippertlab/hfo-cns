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

    image_path = "/data/p_02718/Images/TimeFrequencyPlots_ExploratoryAnalysis_Cortical_/"
    os.makedirs(image_path, exist_ok=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, 1)

    subjects = np.arange(1, 37)
    sfreq = 5000
    cond_names = ['median', 'tibial']

    freqs = np.arange(350., 900., 3.)
    fmin, fmax = freqs[[0, -1]]

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for cond_name in cond_names:  # Conditions (median, tibial)
        power_induced_list = []
        power_evoked_list = []

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
            # Evoked Power - Get evoked and then compute power
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked.reorder_channels(eeg_chans)
            evoked = evoked.pick_channels(channel)
            if channel[0] in evoked.info['bads']:
                evoked.info['bads'] = []
                bad_flag = True
            power_evoked = mne.time_frequency.tfr_stockwell(evoked, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)
            power_evoked_list.append(power_evoked)

            # Induced Power - Get single trial power and then average
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline), preload=True)
            epochs.reorder_channels(eeg_chans)
            epochs = epochs.pick_channels(channel)
            if channel[0] in evoked.info['bads']:
                epochs.info['bads'] = []
                bad_flag = True
            power_induced = mne.time_frequency.tfr_stockwell(epochs, fmin=fmin, fmax=fmax, width=3.0, n_jobs=5)
            # tfr automatically averages after computing each trials TFR
            power_induced_list.append(power_induced)

            # Generate Single Subject Images
            if cond_name == 'tibial':
                tmin = 0.0
                tmax = 0.07
            else:
                tmin = 0.0
                tmax = 0.05
            for type in ['ratio', 'mean']:
                fig, ax = plt.subplots(1, 2)
                power_evoked.plot(picks=channel, baseline=iv_baseline, mode=type, cmap='jet',
                           axes=ax[0], show=False, colorbar=True, dB=False,
                           tmin=tmin, tmax=tmax, vmin=0)
                power_induced.plot(picks=channel, baseline=iv_baseline, mode=type, cmap='jet',
                                  axes=ax[1], show=False, colorbar=True, dB=False,
                                  tmin=tmin, tmax=tmax, vmin=0)
                im = ax[0].images
                cb = im[-1].colorbar
                cb.set_label('Amplitude')
                im = ax[1].images
                cb = im[-1].colorbar
                cb.set_label('Amplitude')
                if bad_flag is True:
                    plt.suptitle(f"Subject {subject} TFR\n"
                              f"Condition: {trigger_name}, Bad Channel")
                else:
                    plt.suptitle(f"Subject {subject} TFR\n"
                              f"Condition: {trigger_name}")
                ax[0].set_title('Evoked Power')
                ax[1].set_title('Induced Power')
                fname = f"{subject_id}_{trigger_name}_{type}"
                plt.tight_layout()
                fig.savefig(image_path + fname + '.png')
                # plt.savefig(image_path + fname + '.pdf', bbox_inches='tight', format="pdf")
                plt.close()

        averaged_evoked = mne.grand_average(power_evoked_list, interpolate_bads=False, drop_bads=False)
        averaged_induced = mne.grand_average(power_induced_list, interpolate_bads=False, drop_bads=False)
        for type in ['mean', 'ratio']:
            fig, ax = plt.subplots(1, 2)
            averaged_evoked.plot(picks=channel, baseline=iv_baseline, mode=type, cmap='jet',
                          axes=ax[0], show=False, colorbar=True, dB=False,
                          tmin=tmin, tmax=tmax, vmin=0)
            averaged_induced.plot(picks=channel, baseline=iv_baseline, mode=type, cmap='jet',
                                 axes=ax[1], show=False, colorbar=True, dB=False,
                                 tmin=tmin, tmax=tmax, vmin=0)
            im = ax[0].images
            cb = im[-1].colorbar
            cb.set_label('Amplitude')
            im = ax[1].images
            cb = im[-1].colorbar
            cb.set_label('Amplitude')
            plt.title(f"Grand Average TFR\n"
                      f"Condition: {trigger_name}")
            ax[0].set_title('Evoked Power')
            ax[1].set_title('Induced Power')
            fname = f"GA_{trigger_name}_{type}"
            plt.tight_layout()
            fig.savefig(image_path+fname+'.png')
            # plt.savefig(image_path+fname+'.pdf', bbox_inches='tight', format="pdf")
            plt.close()