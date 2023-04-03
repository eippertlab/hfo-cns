# Import necessary packages
import mne
from Common_Functions.get_conditioninfo import *
from Common_Functions.get_channels import *
from scipy.io import loadmat
import os
import glob
import numpy as np
import pandas as pd
from Common_Functions import evoked_from_raw
import matplotlib.pyplot as plt
from Common_Functions.pchip_interpolation import PCHIP_interpolation

if __name__ == '__main__':
    srmr_nr = 1
    subjects = np.arange(1, 6)
    # subjects = [1]
    conditions = [2, 3]
    use_repaired = True

    sampling_rate_og = 10000
    montage_path = '/data/pt_02068/cfg/'
    montage_name = 'standard-10-5-cap385_added_mastoids.elp'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    notch_low = df.loc[df['var_name'] == 'notch_freq_low', 'var_value'].iloc[0]
    notch_high = df.loc[df['var_name'] == 'notch_freq_high', 'var_value'].iloc[0]

    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    for subject in subjects:
        for condition in conditions:
            # Set paths
            subject_id = f'sub-{str(subject).zfill(3)}'
            input_path = f"/data/pt_02718/tmp_data/imported/{subject_id}/"

            save_path = f"/data/p_02718/Images/Bipolar_InitialImages/{subject_id}/"
            os.makedirs(save_path, exist_ok=True)
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            stimulation = condition - 1

            if condition == 2:
                channel_names = ['Biceps', 'EP']
            elif condition == 3:
                channel_names = ['KneeM']

            if use_repaired:
                fname = f'bipolar_repaired_{cond_name}.fif'
            else:
                print('Error - Must use data with repaired stimulation artefact')
            raw = mne.io.read_raw_fif(input_path + fname, preload=True)

            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline))
            evoked = epochs.average(picks='all')

            # evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)

            for channel in channel_names:
                fig, ax = plt.subplots(1, 2)
                ax[0].plot(evoked.times, evoked.get_data(picks=[channel]).reshape(-1))
                ax[0].set_title('Wideband')
                ax[0].set_xlim([-20/1000, 60/1000])
                # ax[0].set_ylim([-6e-6, 2e-6])
                # ax[0].axvline(x=0.006, color='red')
                ax[1].plot(evoked.times, evoked.copy().filter(l_freq=400, h_freq=800, n_jobs=len(raw.ch_names), method='iir',
                                                       iir_params={'order': 2, 'ftype': 'butter'}, phase='zero').get_data(picks=[channel]).reshape(-1))
                ax[1].set_title('High Pass Filtered (400-800Hz)')
                ax[1].set_xlim([-20 / 1000, 60 / 1000])
                # ax[1].set_ylim([-1e-7, 1e-7])
                # ax[1].axvline(x=0.006, color='red')
                plt.suptitle(f'Subject {subject}, {trigger_name}, {channel}')
                plt.tight_layout()
                # plt.show()
                plt.savefig(save_path + f'{channel}.png')
