# Run auto method of ICA to remove eog artifacts

import os
import mne
import matplotlib.pyplot as plt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import *


def run_ica(subject, condition, srmr_nr, sampling_rate, choose_limited):

    # Set paths
    subject_id = f'sub-{str(subject).zfill(3)}'
    input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"  # Taking prepared data
    save_path = "/data/pt_02718/tmp_data/ica_cleaned/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)
    cfg_path = "/data/pt_02718/"  # Contains important info about experiment

    # Get the condition information based on the condition read in
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve

    # load cleaned ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif'
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # make a copy to filter
    raw_filtered = raw.copy().drop_channels(['EOGH', 'EOGV'])

    # filtering
    raw_filtered.filter(l_freq=1, h_freq=45, n_jobs=len(raw.ch_names), method='iir',
                        iir_params={'order': 2, 'ftype': 'butter'}, phase='zero')

    # ICA
    ica = mne.preprocessing.ICA(n_components=len(raw_filtered.ch_names), max_iter='auto', random_state=97)
    ica.fit(raw_filtered)

    raw.load_data()

    # Automatically choose ICA components
    ica.exclude = []
    # find which ICs match the ECG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='ECG')
    ica.exclude = eog_indices

    # Just for visualising
    ica.plot_overlay(raw.copy().drop_channels(['E0GV', 'EOGH']), exclude=eog_indices, picks='eeg')
    print(ica.exclude)
    ica.plot_scores(eog_scores)

    # Apply the ica we got from the filtered data onto the unfiltered raw
    ica.apply(raw)

    # Save raw data
    # fname = 'clean_baseline_ica_auto_' + cond_name + '.fif'
    # raw.save(os.path.join(save_path, fname), fmt='double', overwrite=True)
