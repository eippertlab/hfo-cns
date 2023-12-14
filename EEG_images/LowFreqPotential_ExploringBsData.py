# Read in data Birgits data that has not been cleaned - no rereferencing so will be to right mastoid as in og recordings

import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    srmr_nr = 1  # Set the experiment number
    show_each_subject = False  # If we want to look at each subject as generated

    if srmr_nr == 1:
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [3, 2]  # Conditions of interest

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
        conditions = [5, 3]  # Conditions of interest - med_mixed and tib_mixed

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    for condition in conditions:
        evoked_list = []
        for subject in subjects:
            # Set variables
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            subject_id = f'sub-{str(subject).zfill(3)}'

            # Select the right files
            if srmr_nr == 1:
                input_path = f"/data/pt_02718/tmp_data/imported/{subject_id}/"
                fname = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"
                figure_path = '/data/p_02718/Images/EEG/LowFrequency_BeforeCleaning/'
                os.makedirs(figure_path, exist_ok=True)

            elif srmr_nr == 2:
                input_path = f"/data/pt_02718/tmp_data/imported/{subject_id}/"
                fname = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"
                figure_path = '/data/p_02718/Images_2/EEG/LowFrequency_BeforeCleaning/'
                os.makedirs(figure_path, exist_ok=True)

            eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

            raw = mne.io.read_raw_fif(input_path + fname, preload=True)

            # Set montage
            montage_path = '/data/pt_02718/'
            montage_name = 'electrode_montage_eeg_10_5.elp'
            montage = mne.channels.read_custom_montage(montage_path + montage_name)
            raw.set_montage(montage, on_missing="ignore")
            idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
            res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

            # Get correct channels
            if cond_name in ['median', 'med_mixed']:
                channel = 'CP4'
            elif cond_name in ['tibial', 'tib_mixed']:
                channel = 'Cz'

            # now create epochs based on the trigger names
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)

            evoked = epochs.average()
            evoked.reorder_channels(eeg_chans)
            evoked_list.append(evoked)

            # Plot time course
            relevant_ch = evoked.copy().pick_channels([channel])
            fig, ax = plt.subplots(1, 1)
            ax.plot(relevant_ch.times, relevant_ch.data[0, :] * 10 ** 6)
            ax.set_ylabel('Amplitude (\u03BCV)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{subject_id} Time Course')
            if cond_name in ['median', 'med_mixed']:
                ax.set_xlim([0.00, 0.05])
            elif cond_name in ['tibial', 'tib_mixed']:
                ax.set_xlim([0.00, 0.07])
            plt.savefig(figure_path + f'{subject_id}_Time_{cond_name}')
            if show_each_subject:
                plt.show()

            # Plot Spatial Topographies
            if cond_name in ['median', 'med_mixed']:
                times = [0.012, 0.013, 0.014, 0.015, 0.016]
                # times = [0.0188]
            elif cond_name in ['tibial', 'tib_mixed']:
                times = [0.028, 0.029, 0.030, 0.031, 0.032]
                # times = [0.044]
            evoked.plot_topomap(times=times, average=None, ch_type=None, scalings=None, proj=False,
                                  sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                  outlines='head', sphere=None, image_interp='cubic', extrapolate='auto', border='mean',
                                  res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None, cnorm=None,
                                  colorbar=True, cbar_fmt='%3.1f', units=None, axes=None, time_unit='s',
                                  time_format=None, title=f'{subject_id} Spatial Pattern',
                                  nrows=1, ncols='auto', show=False)
            plt.savefig(figure_path + f'{subject_id}_Spatial_{cond_name}')

        # Plot time course
        averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
        relevant_channel = averaged.copy().pick_channels([channel])
        fig, ax = plt.subplots(1, 1)
        ax.plot(relevant_channel.times, relevant_channel.data[0, :] * 10 ** 6)
        ax.set_ylabel('Amplitude (\u03BCV)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Grand Average Time Course, n={len(subjects)}')
        if cond_name in ['median', 'med_mixed']:
            ax.set_xlim([-0.025, 0.065])
        elif cond_name in ['tibial', 'tib_mixed']:
            ax.set_xlim([-0.025, 0.085])
        plt.savefig(figure_path+f'GA_Time_{cond_name}')

        # Plot Spatial Topographies
        if cond_name in ['median', 'med_mixed']:
            times = [0.012, 0.013, 0.014, 0.015, 0.016]
            # times = [0.0188]
        elif cond_name in ['tibial', 'tib_mixed']:
            times = [0.028, 0.029, 0.030, 0.031, 0.032]
        averaged.plot_topomap(times=times, average=None, ch_type=None, scalings=None, proj=False,
                              sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                              outlines='head', sphere=None, image_interp='cubic', extrapolate='auto', border='mean',
                              res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None, cnorm=None,
                              colorbar=True, cbar_fmt='%3.1f', units=None, axes=None, time_unit='s',
                              time_format=None, title=f'Grand Average Spatial Pattern, n={len(subjects)}',
                              nrows=1, ncols='auto', show=False)
        plt.savefig(figure_path+f'GA_Spatial_{cond_name}')