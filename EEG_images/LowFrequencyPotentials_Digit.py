# Read in data Birgit has cleaned up in MATLAB for the low-frequency cortical responses for exp 1 and 2
# Just using this to get the timing of the potentials for future shifting


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    srmr_nr = 2  # Set the experiment number
    show_each_subject = True  # If we want to look at each subject as generated

    subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
    conditions = [2, 4]  # Conditions of interest - med_dig and tib_dig

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
            trigger_names = cond_info.trigger_name
            subject_id = f'sub-{str(subject).zfill(3)}'
            input_path = "/data/pt_02151/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
            fname = f"cnt_clean_{cond_name}.set"
            figure_path = '/data/p_02718/Images_2/EEG/LowFrequency_Digits/'
            os.makedirs(figure_path, exist_ok=True)

            eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

            raw = mne.io.read_raw_eeglab(input_path + fname, preload=True)

            # Set montage
            montage_path = '/data/pt_02718/'
            montage_name = 'electrode_montage_eeg_10_5.elp'
            montage = mne.channels.read_custom_montage(montage_path + montage_name)
            raw.set_montage(montage, on_missing="ignore")
            idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
            res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

            # Get correct channels
            if cond_name == 'med_digits':
                channel = 'CP4'
            elif cond_name == 'tib_digits':
                channel = 'Cz'

            # now create epochs based on the trigger names
            events, event_ids = mne.events_from_annotations(raw)
            fig, ax = plt.subplots(1, 1)
            fig2, ax2 = plt.subplots(1, 2, width_ratios=[20, 1])
            for trigger_name in trigger_names:
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                    baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)

                evoked = epochs.average()
                evoked.reorder_channels(eeg_chans)
                evoked_list.append(evoked)

                # Plot time course
                relevant_ch = evoked.copy().pick_channels([channel])
                ax.plot(relevant_ch.times, relevant_ch.data[0, :] * 10 ** 6, label=trigger_name)
                ax.set_ylabel('Amplitude (\u03BCV)')
                ax.set_xlabel('Time (s)')
                ax.set_title(f'{subject_id} Time Course')
                if cond_name == 'med_digits':
                    ax.set_xlim([0.00, 0.05])
                elif cond_name == 'tib_digits':
                    ax.set_xlim([0.00, 0.07])

                # Plot Spatial Topographies
                if cond_name == 'med_digits':
                    times = [0.022]
                elif cond_name == 'tib_digits':
                    times = [0.047]
                evoked.plot_topomap(times=times, average=None, ch_type=None, scalings=None, proj=False,
                                      sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                      outlines='head', sphere=None, image_interp='cubic', extrapolate='auto', border='mean',
                                      res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None, cnorm=None,
                                      colorbar=True, cbar_fmt='%3.1f', units=None, axes=ax2, time_unit='s',
                                      time_format=None, title=f'{subject_id} Spatial Pattern',
                                      nrows=1, ncols='auto', show=False)

            fig.legend()
            fig.tight_layout()
            fig2.tight_layout()
            if show_each_subject:
                plt.show()
            fig.savefig(figure_path + f'{subject_id}_Time_{cond_name}')
            fig2.savefig(figure_path + f'{subject_id}_Spatial_{cond_name}')
            plt.close()