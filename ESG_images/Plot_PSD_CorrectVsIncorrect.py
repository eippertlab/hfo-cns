# Script to plot the power spectral density of the grand average data in the correct and incorrect patch

import mne
import os
import numpy as np
from scipy.io import loadmat
from Common_Functions.evoked_from_raw import evoked_from_raw
import matplotlib.pyplot as plt


if __name__ == '__main__':
    subjects = np.arange(1, 37)  # 1 through 36 to access subject data
    cond_names = ['median', 'tibial']
    sampling_rate = 5000
    full = True  # Do up to 2000Hz, otherwise up to 400Hz

    cfg_path = "/data/pt_02569/"  # Contains important info about experiment
    cfg = loadmat(cfg_path + 'cfg.mat')
    notch_freq = cfg['notch_freq'][0]
    esg_bp_freq = cfg['esg_bp_freq'][0]

    iv_epoch = cfg['iv_epoch'][0] / 1000
    iv_baseline = cfg['iv_baseline'][0] / 1000

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02718/Images/BeforeCCA_SpatialSpecificity/PSD/"
    os.makedirs(image_path, exist_ok=True)

    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_correct = []
        evoked_list_incorrect = []

        if cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'
            correct_channel = ['L1']
            incorrect_channel = ['SC6']

        elif cond_name == 'median':
            trigger_name = 'Median - Stimulation'
            correct_channel = ['SC6']
            incorrect_channel = ['L1']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            input_path = f"/data/pt_02718/tmp_data/ssp_cleaned/{subject_id}" + "/"
            raw = mne.io.read_raw_fif(f"{input_path}ssp6_cleaned_{cond_name}.fif", preload=True)
            evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
            evoked.reorder_channels(esg_chans)
            evoked_list_correct.append(evoked.copy().pick_channels(correct_channel))
            evoked_list_incorrect.append(evoked.copy().pick_channels(incorrect_channel))

        averaged_correct = mne.grand_average(evoked_list_correct, interpolate_bads=False, drop_bads=False)
        averaged_incorrect = mne.grand_average(evoked_list_incorrect, interpolate_bads=False, drop_bads=False)

        correct_data = np.mean(averaged_correct.data[:, :], axis=0)
        incorrect_data = np.mean(averaged_incorrect.data[:, :], axis=0)

        plt.figure()
        plt.psd(correct_data, NFFT=512, Fs=sampling_rate, label='Correct Channel')
        plt.psd(incorrect_data, NFFT=512, Fs=sampling_rate, label='Incorrect Channel')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD (dB/Hz)')
        if cond_name == 'median':
            plt.ylim([-225, -180])
        elif cond_name == 'tibial':
            plt.ylim([-220, -170])
            # plt.yticks(np.arange(-240, -130, 20.0))
        if full:
            plt.xlim([0, 2000])
            fname = f"{trigger_name}_full.png"
        else:
            plt.xlim([0, 400])
            fname = f"{trigger_name}.png"
        plt.legend()

        plt.title(f"Condition: {trigger_name}")
        plt.savefig(image_path+fname)
        plt.close()
