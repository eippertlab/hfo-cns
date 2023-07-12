# Script to plot the time-course of the cardiac activity in high frequency zone before and after SSP

import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_conditioninfo import get_conditioninfo
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 2

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        sfreq = 5000
        conditions = [2, 3]
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        sfreq = 5000
        conditions = [3, 5]

    # Baseline is the 100ms period before the artefact occurs
    iv_baseline = [-300 / 1000, -200 / 1000]
    # Want 200ms before and 400ms after the R-peak in our epoch - need baseline outside this
    iv_epoch = [-300 / 1000, 400 / 1000]

    esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                 'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                 'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                 'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                 'S23']

    image_path = "/data/p_02718/Images/HighFrequencyCardiac/"
    os.makedirs(image_path, exist_ok=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for condition in conditions:
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        evoked_list_imported = []
        evoked_list_ssp = []

        if cond_name in ['tibial', 'tib_mixed']:
            full_name = 'Lumbar Spinal Cord'
            trigger_name = 'qrs'
            channel = ['L1']

        elif cond_name in ['median', 'med_mixed']:
            full_name = 'Cervical Spinal Cord'
            trigger_name = 'qrs'
            channel = ['SC6']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'

            if srmr_nr == 1:
                input_path_ssp = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
                fname_ssp = f"ssp6_cleaned_{cond_name}.fif"
                input_path_imported = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
                fname_imported = f"noStimart_sr{sfreq}_{cond_name}_withqrs.fif"
            else:
                input_path_ssp = "/data/pt_02718/tmp_data_2/ssp_cleaned/" + subject_id + "/"
                fname_ssp = f"ssp6_cleaned_{cond_name}.fif"
                input_path_imported = "/data/pt_02718/tmp_data_2/imported/" + subject_id + "/"
                fname_imported = f"noStimart_sr{sfreq}_{cond_name}_withqrs.fif"

            for input_path, fname in zip([input_path_imported, input_path_ssp], [fname_imported, fname_ssp]):
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                raw.filter(l_freq=400, h_freq=800, n_jobs=len(raw.ch_names), method='iir',
                           iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked = evoked.copy().pick_channels(channel).crop(tmin=-0.2, tmax=0.4)
                if input_path == input_path_imported:
                    evoked_list_imported.append(evoked)
                else:
                    evoked_list_ssp.append(evoked)

        fig, ax = plt.subplots()
        averaged_imported = mne.grand_average(evoked_list_imported, interpolate_bads=False, drop_bads=False)
        averaged_ssp = mne.grand_average(evoked_list_ssp, interpolate_bads=False, drop_bads=False)
        ax.plot(averaged_imported.times, averaged_imported.data[0, :] * 10 ** 6, label='Before SSP')
        ax.plot(averaged_ssp.times, averaged_ssp.data[0, :] * 10 ** 6, label='After SSP')

        ax.set_ylabel('Amplitude (\u03BCV)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f"{full_name}")
        image_fname = f"Dataset{srmr_nr}_{full_name}.png"
        plt.legend()
        plt.tight_layout()
        plt.savefig(image_path + image_fname)
        plt.clf()
