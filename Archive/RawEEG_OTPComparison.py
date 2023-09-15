import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    subjects = [6, 15, 18, 25]
    sfreq = 5000
    cond_names = ['median']

    plot_single_subject = True
    plot_grand_average = True


    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list = []

        if cond_name == 'tibial':
            full_name = 'Tibial Nerve Stimulation'
            trigger_name = 'Tibial - Stimulation'
            channel = ['Cz']

        elif cond_name == 'median':
            full_name = 'Median Nerve Stimulation'
            trigger_name = 'Median - Stimulation'
            channel = ['CP4']

        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'

            before_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
            after_path = "/data/pt_02718/tmp_data_otp/otp_cleaned_eeg/" + subject_id + "/"
            before_fname = f'noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif'
            after_fname = f'otp_cleaned_{cond_name}.fif'

            raw_before = mne.io.read_raw_fif(before_path + before_fname, preload=True)
            raw_after = mne.io.read_raw_fif(after_path + after_fname, preload=True)

            montage_path = '/data/pt_02718/'
            montage_name = 'electrode_montage_eeg_10_5.elp'
            montage = mne.channels.read_custom_montage(montage_path + montage_name)
            raw_before.set_montage(montage, on_missing="ignore")
            raw_after.set_montage(montage, on_missing="ignore")

            # # Plot all channels
            # fig, ax = plt.subplots(1, 2)
            # titles = ['Raw', 'OTP']
            # count = 0
            # for raw in [raw_before, raw_after]:
            #     events, event_ids = mne.events_from_annotations(raw)
            #     event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            #     epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
            #                         baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)
            #     evoked = epochs.average()
            #
            #     evoked.plot(picks=None, exclude='bads', unit=True, show=False, ylim=None, xlim='tight',
            #                 titles=titles[count], axes=ax[count])
            #     ax[count].set_xlim([-0.025, 0.065])
            #     ax[count].set_ylim([-0.3, 0.3])
            #     count+=1
            # plt.tight_layout()
            # plt.show()

            # Plot channel of interest
            fig, ax = plt.subplots()
            titles = ['Raw', 'OTP']
            for raw, title in zip([raw_before, raw_after], titles):
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                                    baseline=tuple(iv_baseline), preload=True, reject_by_annotation=True)
                evoked = epochs.average()

                ax.plot(evoked.times, evoked.pick_channels(channel).get_data().reshape(-1)*10**6, label=title)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('uV')
                ax.set_xlim([-0.025, 0.065])
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'/data/p_02718/Images_OTP/EEG/InterestChannel_Evoked/{subject_id}_{cond_name}_{channel[0]}')