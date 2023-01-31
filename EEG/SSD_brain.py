# Script to actually run SSD on the cortical data


import os
import mne
import numpy as np
from mne.decoding import SSD
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


if __name__ == '__main__':
    conditions = [2, 3]
    srmr_nr = 1
    subjects = np.arange(1, 3)
    freq_band = 'sigma'
    sfreq = 5000

    for condition in conditions:
        for subject in subjects:
            # Set variables
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            subject_id = f'sub-{str(subject).zfill(3)}'

            cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
            df = pd.read_excel(cfg_path)
            iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                           df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
            iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                        df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

            # Select the right files
            input_path = f"/data/pt_02718/tmp_data/imported/{subject_id}" + "/"
            raw = mne.io.read_raw_fif(f"{input_path}noStimart_sr5000_{cond_name}_withqrs_eeg.fif", preload=True)
            save_path = "/data/pt_02718/tmp_data/ssd_eeg/" + subject_id + "/"
            os.makedirs(save_path, exist_ok=True)

            eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

            # Set montage
            montage_path = '/data/pt_02718/'
            montage_name = 'electrode_montage_eeg_10_5.elp'
            montage = mne.channels.read_custom_montage(montage_path + montage_name)
            raw.set_montage(montage, on_missing="ignore")
            idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
            res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

            raw = raw.pick_channels(eeg_chans)
            # now create epochs based on the trigger names
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                                baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)

            if cond_name == 'median':
                sep_latency = 20
            elif cond_name == 'tibial':
                sep_latency = 40
            else:
                print('Invalid condition name attempted for use')
                exit()

            freqs_sig = 500, 600
            freqs_noise = 450, 650

            #########################################################################################################
            # Working with Raw
            #########################################################################################################
            # raw.crop(50., 110.).load_data()  # crop for memory purposes

            # Stores all the options we want applied in the SSD
            ssd = SSD(info=raw.info,
                      reg='oas',
                      sort_by_spectral_ratio=True,
                      filt_params_signal=dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                                              l_trans_bandwidth=1, h_trans_bandwidth=1, n_jobs=12),
                      filt_params_noise=dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                                             l_trans_bandwidth=1, h_trans_bandwidth=1, n_jobs=12),
                      n_components=4, rank='full')
            # Performs the actual fit - estimates the SSD decomposition
            ssd.fit(X=raw.get_data())
            # ssd.fit(X=epochs.get_data())

            # Removes the selected components from the signal - leaves us only the 4 we asked for
            ssd_sources = ssd.transform(X=epochs.get_data())  # Returns X_SSD

            # patterns_ has shape (n_components, n_channels)
            pattern = mne.EvokedArray(data=ssd.patterns_[:4].T,
                                      info=ssd.info)
            pattern.plot_topomap(units=dict(eeg='A.U.'), time_format='', cmap='jet')

            # filters_ has shape (n_channels, n_components)
            # epochs have shape (n_epochs, n_channels, n_times)
            # Want to apply it to the data band pass filtered from 400-800
            # epochs_filtered = epochs.filter(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
            #                                 l_trans_bandwidth=1, h_trans_bandwidth=1, n_jobs=12)
            # filtered_trials = np.tensordot(ssd.filters_[:, :4], epochs_filtered.get_data(), axes=(0, 1))
            fig, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            # print(np.shape(epochs.times))
            # print(np.shape(filtered_trials.mean(axis=1).reshape(-1)))
            for i in np.arange(0, 4):
                # axes[i].plot(epochs.times, filtered_trials.mean(axis=1)[i, :])
                axes[i].plot(epochs.times, ssd_sources.mean(axis=0)[i, :])
                axes[i].set_xlim([-0.02, 0.07])
                axes[i].set_title(f'Component {i+1}')
            plt.show()

            # # Get psd of SSD-filtered signals.
            # psd, freqs = mne.time_frequency.psd_array_welch(
            #     ssd_sources, sfreq=raw.info['sfreq'], n_fft=4096)
            #
            # around600 = freqs<1000
            # # for highlighting the freq. band of interest
            # bandfilt = (freqs_sig[0] <= freqs) & (freqs <= freqs_sig[1])
            # fig, ax = plt.subplots(1)
            # ax.loglog(freqs[around600], psd[0, around600], label='max SNR')
            # ax.loglog(freqs[around600], psd[-1, around600], label='min SNR')
            # ax.loglog(freqs[around600], psd[:, around600].mean(axis=0), label='mean')
            # ax.fill_between(freqs[bandfilt], 0, 10000, color='green', alpha=0.15)
            # ax.set_xlabel('log(frequency)')
            # ax.set_ylabel('log(power)')
            # ax.legend()

            plt.show()

            exit()
