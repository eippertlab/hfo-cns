# Plot grand average time courses and spatial patterns after application of bCSTP


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
from meet._cSPoC import pattern_from_filter


if __name__ == '__main__':
    subjects = np.arange(1, 7)
    conditions = [2, 3]
    freq_bands = ['sigma', 'kappa']
    srmr_nr = 1
    sfreq = 5000

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    # Get a raw file so I can use the montage
    raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data/freq_banded_eeg/sub-001/sigma_median.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)
    figure_path = '/data/p_02718/Images/bCSTP_eeg/GrandAverage/'
    os.makedirs(figure_path, exist_ok=True)

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list = []
            spatial_pattern = []
            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                ##########################################################
                # Time  Course Information
                ##########################################################
                # Select the right files
                input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
                fname = f"{freq_band}_{cond_name}.fif"
                eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)

                # Set montage
                montage_path = '/data/pt_02718/'
                montage_name = 'electrode_montage_eeg_10_5.elp'
                montage = mne.channels.read_custom_montage(montage_path + montage_name)
                raw.set_montage(montage, on_missing="ignore")
                idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
                res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

                # create c+ epochs from -40ms to 90ms
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epochs_pos = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                        baseline=tuple(iv_baseline), preload=True)
                # epochs_pos.set_eeg_reference(ref_channels='average')
                epochs_pos = epochs_pos.pick_channels(ch_names=eeg_chans)
                data_pos = np.transpose(epochs_pos.get_data(), (1, 2, 0))

                # Read in spatial filters
                file_path = f'/data/pt_02718/tmp_data/bCSTP_eeg/{subject_id}/'
                with open(f'{file_path}W_{freq_band}_{cond_name}.pkl', 'rb') as f:
                    W = pickle.load(f)
                keeps = 1
                spatial_filters = W[-1][:, 0:keeps]  # Keep top spatial filters

                # Spatially filter trials & get top spatial pattern
                spatially_filtered_trials = np.tensordot(spatial_filters, data_pos, axes=(0, 0))
                spatial_patterns = pattern_from_filter(spatial_filters, data_pos)

                #  Epoch data class for the spatially filtered
                events = epochs_pos.events
                event_id = epochs_pos.event_id
                tmin = iv_epoch[0]
                sfreq = sfreq

                ch_names = [f'Component {i + 1}' for i in np.arange(0, keeps)]
                ch_types = ['eeg' for i in np.arange(0, len(ch_names))]

                # Initialize an info structure
                info_full_filt = mne.create_info(
                    ch_names=ch_names,
                    ch_types=ch_types,
                    sfreq=sfreq
                )

                # Create and save
                bcstp_epochs_spat = mne.EpochsArray(np.transpose(spatially_filtered_trials, axes=(2, 0, 1)),
                                                    info_full_filt,
                                                    events, tmin,
                                                    event_id)
                bcstp_epochs_spat = bcstp_epochs_spat.apply_baseline(baseline=tuple(iv_baseline))

                # Need to pick channel based on excel sheet
                channel = 'Component 1'
                epochs = bcstp_epochs_spat.pick_channels([channel])
                evoked = epochs.average()
                data = evoked.data
                evoked_list.append(data)

                ############################################################
                # Spatial Pattern Extraction
                ############################################################
                spatial_pattern.append(spatial_patterns[0, :])

            # Get grand average across chosen epochs, and spatial patterns
            grand_average = np.mean(evoked_list, axis=0)
            grand_average_spatial = np.mean(spatial_pattern, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(epochs.times, grand_average[0, :])
            ax.set_ylabel('Cleaned SEP Amplitude (AU)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Time Course, n={len(subjects)}')
            if cond_name == 'median':
                ax.set_xlim([0.01, 0.05])
            else:
                ax.set_xlim([0.03, 0.07])

            plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}')

            # Plot Spatial Pattern
            fig, ax = plt.subplots(1, 1)
            chan_labels = epochs.ch_names
            mne.viz.plot_topomap(data=grand_average_spatial, pos=res, ch_type='eeg', sensors=True, names=None,
                                 contours=6, outlines='head', sphere=None, image_interp='cubic',
                                 extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                 cnorm=None, axes=ax, show=False)
            ax.set_title(f'Grand Average Spatial Pattern, n={len(subjects)}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cb = fig.colorbar(ax.images[-1], cax=cax, shrink=0.6, orientation='vertical')
            plt.savefig(figure_path+f'GA_Spatial_{freq_band}_{cond_name}')
