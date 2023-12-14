# Script to plot the time-frequency decomposition in dB about the spinal triggers
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.appy_cca_weights import apply_cca_weights
from Common_Functions.get_channels import get_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
import pickle
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    image_path = "/data/p_02718/Images/TimeFrequencyPlots_ExploratoryAnalysis_Cortical_CCAtoBroadband/"
    os.makedirs(image_path, exist_ok=True)

    subjects = np.arange(1, 37)
    sfreq = 5000
    freq_band = 'sigma'
    cond_names = ['median', 'tibial']
    folder = 'tmp_data'
    conditions = [2, 3]

    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    freqs = np.arange(350., 900., 3.)
    fmin, fmax = freqs[[0, -1]]

    # Cortical Excel files
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for condition in conditions:  # Conditions (median, tibial)
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        power_induced_list = []
        power_evoked_list = []

        if cond_name == 'tibial':
            full_name = 'Tibial Nerve Stimulation'
            trigger_name = 'Tibial - Stimulation'

        elif cond_name == 'median':
            full_name = 'Median Nerve Stimulation'
            trigger_name = 'Median - Stimulation'

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            df = df_cortical
            with open(f'/data/pt_02718/{folder}/cca_eeg/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                      'rb') as f:
                W_st = pickle.load(f)

            fname = f"noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif"
            input_path = f"/data/pt_02718/{folder}/imported/{subject_id}/"
            raw_data = mne.io.read_raw_fif(input_path + fname, preload=True)

            events, event_ids = mne.events_from_annotations(raw_data)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw_data, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline), preload=True)

            apply_weights_kwargs = dict(
                weights=W_st
            )
            cor_names = [f'Cor{i}' for i in np.arange(1, len(W_st) + 1)]
            epochs.pick(eeg_chans).reorder_channels(eeg_chans)
            epochs_ccafiltered = epochs.apply_function(apply_cca_weights, picks=eeg_chans,
                                                       channel_wise=False, **apply_weights_kwargs)
            channel_map = {eeg_chans[i]: cor_names[i] for i in range(len(eeg_chans))}
            epochs_ccafiltered.rename_channels(channel_map)
            epochs_ccafiltered.crop(tmin=-0.06, tmax=0.1)
            evoked_ccafiltered = epochs_ccafiltered.average()

            channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
            channel = f'Cor{channel_no}'
            inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]

            if channel_no != 0:  # 0 marks subjects where no component is selected
                evoked_ccafiltered.pick(channel)
                epochs_ccafiltered.pick(channel)
                # Renaming them all to Cor1 so we can do grand average - since we have already picked the right
                # channel, we're just renaming it
                evoked_ccafiltered.rename_channels({channel: 'Cor1'}, allow_duplicates=False, verbose=None)
                epochs_ccafiltered.rename_channels({channel: 'Cor1'}, allow_duplicates=False, verbose=None)

                # Induced Power - Get single trial power and then average
                power_induced = mne.time_frequency.tfr_stockwell(epochs_ccafiltered, fmin=fmin, fmax=fmax, width=3.0,
                                                                 n_jobs=5)

                # Evoked Power - Get evoked and then compute power
                power_evoked = mne.time_frequency.tfr_stockwell(evoked_ccafiltered, fmin=fmin, fmax=fmax, width=3.0,
                                                         n_jobs=5)

                power_evoked_list.append(power_evoked)
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
                    power_evoked.plot(baseline=iv_baseline, mode=type, cmap='jet',
                               axes=ax[0], show=False, colorbar=True, dB=False,
                               tmin=tmin, tmax=tmax, vmin=0)
                    power_induced.plot(baseline=iv_baseline, mode=type, cmap='jet',
                                      axes=ax[1], show=False, colorbar=True, dB=False,
                                      tmin=tmin, tmax=tmax, vmin=0)
                    im = ax[0].images
                    cb = im[-1].colorbar
                    cb.set_label('Amplitude')
                    im = ax[1].images
                    cb = im[-1].colorbar
                    cb.set_label('Amplitude')
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
            averaged_evoked.plot(baseline=iv_baseline, mode=type, cmap='jet',
                          axes=ax[0], show=False, colorbar=True, dB=False,
                          tmin=tmin, tmax=tmax, vmin=0)
            averaged_induced.plot(baseline=iv_baseline, mode=type, cmap='jet',
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