# Script to actually run CCA on the data
# Using the meet package https://github.com/neurophysics/meet.git to run the CCA


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    srmr_nr = 1
    freq_bands = ['sigma', 'kappa']

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list = []
            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Select the right files based on the data_string
                input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
                fname = f"{freq_band}_{cond_name}.fif"
                figure_path = '/data/p_02718/Images/EEG/GrandAverage/'
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
                if cond_name == 'median':
                    channel = 'CP4'
                elif cond_name == 'tibial':
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
            averaged = mne.grand_average(evoked_list, interpolate_bads=False, drop_bads=False)
            relevant_channel = averaged.copy().pick_channels([channel])
            fig, ax = plt.subplots(1, 1)
            ax.plot(relevant_channel.times, relevant_channel.data[0, :] * 10 ** 6)
            ax.set_ylabel('Amplitude (\u03BCV)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Time Course, n={len(subjects)}')
            if cond_name == 'median':
                ax.set_xlim([0.00, 0.05])
            else:
                ax.set_xlim([0.00, 0.07])
            plt.savefig(figure_path+f'GA_Time_{freq_band}_{cond_name}')

            # Plot Spatial Topographies
            if cond_name == 'median':
                times = [0.016, 0.018, 0.020, 0.022, 0.024]
            elif cond_name == 'tibial':
                times = [0.036, 0.038, 0.040, 0.042, 0.044]
            averaged.plot_topomap(times=times, average=None, ch_type=None, scalings=None, proj=False,
                                  sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                  outlines='head', sphere=None, image_interp='cubic', extrapolate='auto', border='mean',
                                  res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None, cnorm=None,
                                  colorbar=True, cbar_fmt='%3.1f', units=None, axes=None, time_unit='s',
                                  time_format=None, title=f'Grand Average Spatial Pattern, n={len(subjects)}',
                                  nrows=1, ncols='auto', show=True)
            plt.savefig(figure_path+f'GA_Spatial_{freq_band}_{cond_name}')
