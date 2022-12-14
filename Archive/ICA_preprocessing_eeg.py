# ICA pre-processing for EEG data

import matplotlib.pyplot as plt
import mne
import os
import pandas as pd
from Common_Functions.get_channels import get_channels
from Common_Functions.get_conditioninfo import get_conditioninfo

filter_ica = [0.5, 45]
filtermethod_ica = 'iir'
filterorder_ica = 8  # for method = 'iir'
filtertype_ica = 'butter'  # for method = 'iir'
ica_method = 'extended-infomax'


def run_ica_preprocessing(subject, condition, sampling_rate, srmr_nr):
    ica_filter = True
    run_ica = True
    inspect_ica = True
    apply_ica = True

    eeg_chans, esg_chans, bipolar_chans = get_channels(subject_nr=subject, includesEcg=False, includesEog=False,
                                                       study_nr=srmr_nr)
    subject_id = f'sub-{str(subject).zfill(3)}'
    load_path = "/data/pt_02718/tmp_data/imported/" + subject_id
    save_path = "/data/pt_02718/tmp_data/ica_corrected_eeg/" + subject_id
    os.makedirs(save_path, exist_ok=True)
    rerun_ica_path = "/data/pt_02718/tmp_data/ica_corrected_eeg/"

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name

    #############################
    """filter for ICA with butterworth 4th order filter applied as zero-phase"""
    #############################
    if ica_filter:
        data = mne.io.read_raw_fif(os.path.join(load_path, f"noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif"),
                                   preload=True)
        data.filter(filter_ica[0], filter_ica[1], method=filtermethod_ica,
                    iir_params={'order': filterorder_ica, 'ftype': filtertype_ica})  # bandpass filter data
        data.save(os.path.join(save_path, f'filteredICA_{cond_name}.fif'), fmt='double')

    ##############################################
    """run ICA per condition and block, save results"""
    ##############################################
    if run_ica:
        data = mne.io.read_raw_fif(os.path.join(save_path, f'filteredICA_{cond_name}.fif'), preload=True)
        ica = mne.preprocessing.ICA(method=ica_method)  # set up ica
        ica.fit(data)  # fit ICA
        ica.save(os.path.join(save_path, f'ICA_{cond_name}.fif'))

    ###############################################
    """inspect ICA results and mark components to reject"""
    ##############################################
    if inspect_ica:
        raw = mne.io.read_raw_fif(os.path.join(save_path, f'filteredICA_{cond_name}.fif'), preload=True)
        ica = mne.preprocessing.read_ica(os.path.join(save_path, f'ICA_{cond_name}.fif'))
        events, event_ids = mne.events_from_annotations(raw)
        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
        epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                            baseline=tuple(iv_baseline))
        print(events)
        print('Please click on time courses of ICs to reject!')
        ica.plot_components(inst=epochs)  # plot components
        ica.plot_sources(raw, block=True)  # plot their time course, stops until all windows are closed
        ica.save(os.path.join(save_path, f'ICA_withBads_{cond_name}.fif'))

    #####################################################
    """ apply preprocessing
        check ICA solution """
    #####################################################
    if apply_ica:
        breaker = False
        data = mne.io.read_raw_fif(os.path.join(load_path, f"noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif"),
                                   preload=True)

        ica = mne.preprocessing.read_ica(os.path.join(save_path, f'ICA_withBads_{cond_name}.fif'))  # load ICA solution
        events, event_ids = mne.events_from_annotations(data)
        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
        epochs = mne.Epochs(data, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                            baseline=tuple(iv_baseline))
        evoked_pre = epochs.average()  # calculate ERP
        evoked_post = evoked_pre.copy()
        ica.apply(evoked_post)
        for electrode in eeg_chans:
            idx = evoked_pre.info['ch_names'].index(electrode)
            data_pre = evoked_pre.to_data_frame(picks=[idx])
            data_post = evoked_post.to_data_frame(picks=[idx])
            plt.plot(data_pre)  # plot ERP with and without IC rejection
            plt.plot(data_post)
            plt.legend(['Pre ICA', 'Post ICA'])
            plt.title(electrode)
            plt.show(block=True)
        check = False
        while not check:  # wait for user confirmation
            user_input = input("Did ICA work? Press 'y' to confirm and 'n' to add subj to rerun list")
            if user_input == 'y':
                check = True
                bad_subjs_evoked = []
            elif user_input == 'n':
                bad_subjs_evoked = [subject_id]
                check = True
            else:
                print('Wrong input!')

        for electrode in eeg_chans:
            idx = data.info['ch_names'].index(electrode)
            ica.plot_overlay(data, picks=[idx], start=0, stop=data.n_times, title=electrode)
            # think over maximizing options again#
            plt.show(block=True)
        check = False
        while not check:
            user_input = input("Did ICA work? Press 'y' to confirm and 'n' to break")
            if user_input == 'y':
                check = True
                bad_subjs_overlay = []
            elif user_input == 'n':
                check = True
                bad_subjs_overlay = [subject_id]
            else:
                print('Wrong input!')

        # Want to check if bad_subjs_overlay or bad_subjs_evoked are in bad_subjs.txt
        # If not, add them
        filename = rerun_ica_path + 'bad_subjects.txt'
        my_file = open(filename, "r")
        data = my_file.read()
        to_rerun_ica = data.split("\n")
        my_file.close()

        for word in bad_subjs_evoked:
            if word not in to_rerun_ica:
                to_rerun_ica.append(word)
        for word in bad_subjs_overlay:
            if word not in to_rerun_ica:
                to_rerun_ica.append(word)

        with open(filename, mode="w") as outfile:
            for s in to_rerun_ica:
                outfile.write("%s\n" % s)

        ica.apply(data)
        data.save(os.path.join(save_path, f'preprocessed_{cond_name}.fif'), fmt='double')
        fig = data.plot_psd(fmax=250)  # plots spectogram
        plt.title(f'{subject_id}')
        plt.savefig(save_path)
        plt.close()

