import mne
import os
import numpy as np
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    data_type = 'Spinal'  # Spinal or Cortical
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    srmr_nr = 1
    if srmr_nr == 1:
        sfreq = 5000
        cond_names = ['rest']
        subjects = np.arange(1, 7)
        add = ''
    elif srmr_nr == 2:
        sfreq = 5000
        cond_names = ['rest']
        subjects = np.arange(1, 7)
        add = '_2'

    if data_type == 'Spinal':
        image_path_singlesubject = f"/data/p_02718/Images{add}/ESG/RestingState_Test/Evoked/SingleSubject/"
        image_path_grandaverage = f"/data/p_02718/Images{add}/ESG/RestingState_Test/Evoked/GrandAverage/"
    else:
        image_path_singlesubject = f"/data/p_02718/Images{add}/EEG/RestingState_Test/Evoked/SingleSubject/"
        image_path_grandaverage = f"/data/p_02718/Images{add}/EEG/RestingState_Test/Evoked/GrandAverage/"
    os.makedirs(image_path_singlesubject, exist_ok=True)
    os.makedirs(image_path_grandaverage, exist_ok=True)

    # To use mne grand_average method, need to generate a list of evoked potentials for each subject
    for cond_name in cond_names:  # Conditions (median, tibial)
        evoked_list_rest = []
        evoked_list_trig = []
        if srmr_nr == 1:
            cond_name_trig = 'median'
            trigger_name = 'Median - Stimulation'
        elif srmr_nr == 2:
            cond_name_trig = 'med_mixed'
            trigger_name = 'medMixed'
        if data_type == 'Spinal':
            channel = ['SC6']
        else:
            channel = ['CP4']

        for subject in subjects:  # All subjects
            subject_id = f'sub-{str(subject).zfill(3)}'
            eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
            if data_type == 'Spinal':
                input_path = f"/data/pt_02718/tmp_data{add}/freq_banded_esg/{subject_id}/"
            else:
                input_path = f"/data/pt_02718/tmp_data/freq_banded_eeg/{subject_id}/"
            fname_rest = f"sigma_{cond_name}.fif"
            raw_rest = mne.io.read_raw_fif(input_path + fname_rest, preload=True)

            fname = f"sigma_{cond_name_trig}.fif"
            raw_trig = mne.io.read_raw_fif(input_path + fname, preload=True)
            events, event_ids = mne.events_from_annotations(raw_trig)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs_trig = mne.Epochs(raw_trig, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline))
            epochs_rest = mne.Epochs(raw_rest, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                baseline=tuple(iv_baseline))

            fig, ax = plt.subplots(1, 1)
            for epochs, evoked_list in zip([epochs_rest, epochs_trig],
                                           [evoked_list_rest, evoked_list_trig]):
                evoked = epochs.average()
                evoked = evoked.pick_channels(channel)
                evoked_list.append(evoked)

                # Generate Single Subject Images
                tmin = 0.0
                tmax = 0.05
                plt.plot(evoked.times, evoked.get_data().reshape(-1)*10**6)
            ax.set_xlim([tmin, tmax])
            plt.title(f"Subject {subject}")
            fname = f"{subject_id}_{trigger_name}"
            fig.legend(['Resting', 'Median'])
            fig.savefig(image_path_singlesubject + fname + '.png')
            # plt.show()
            # plt.savefig(image_path_singlesubject + fname + '.pdf', bbox_inches='tight', format="pdf")
            plt.close(fig)

        averaged_rest = mne.grand_average(evoked_list_rest, interpolate_bads=False, drop_bads=False)
        averaged_trig = mne.grand_average(evoked_list_trig, interpolate_bads=False, drop_bads=False)
        fig, ax = plt.subplots(1, 1)
        plt.plot(averaged_rest.times, averaged_rest.get_data().reshape(-1) * 10 ** 6)
        plt.plot(averaged_trig.times, averaged_trig.get_data().reshape(-1) * 10 ** 6)
        ax.set_xlim([tmin, tmax])
        plt.title(f"Grand Average, n = {len(evoked_list_rest)}")
        plt.legend(['Resting', 'Median'])
        fname = f"{trigger_name}_n={len(evoked_list_rest)}"
        fig.savefig(image_path_grandaverage+fname+'.png')
        plt.savefig(image_path_grandaverage+fname+'.pdf', bbox_inches='tight', format="pdf")
        plt.close(fig)
