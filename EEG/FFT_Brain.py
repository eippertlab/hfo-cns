# Script to look at spectral peaks of the cortical data


import os
import mne
import numpy as np
from scipy.fft import rfft, rfftfreq
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

            raw = raw.pick_channels(eeg_chans)
            spectrum = raw.compute_psd()
            spectrum.plot(average=False)
            plt.xlim([400, 800])
            plt.show()
            # raw.filter(l_freq=400, h_freq=800, n_jobs=len(raw.ch_names),
            #            method='iir', iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')
            #
            # fourier_transform = np.fft.rfft(raw.get_data().reshape(-1))
            # abs_fourier_transform = np.abs(fourier_transform)
            # power_spectrum = np.square(abs_fourier_transform)
            # frequency_prep = np.linspace(0, sfreq / 2, len(power_spectrum))
            # plt.plot(frequency_prep, power_spectrum, color='black')
            # plt.show()

            # FFT = np.fft.fft(data)
            # new_N = int(len(FFT) / 2)
            # f_nat = 1
            # new_X = np.linspace(10 ** -12, f_nat / 2, new_N, endpoint=True)
            # new_Xph = 1.0 / (new_X)
            # FFT_abs = np.abs(FFT)
            # plt.plot(new_Xph, 2 * FFT_abs[0:int(len(FFT) / 2.)] / len(new_Xph), color='black')
            # plt.xlabel('Period ($h$)', fontsize=20)
            # plt.ylabel('Amplitude', fontsize=20)
            # plt.title('(Fast) Fourier Transform Method Algorithm', fontsize=20)
            # plt.grid(True)
            # plt.xlim(0, 1000)
            # plt.show()
            exit()
