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

def run_icablinkremoval(subject, condition, srmr_nr, sfreq):
    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    if cond_name in ['median', 'med_mixed']:
        channel = 'CP4'
    else:
        channel = 'Cz'

    input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
    fname = f'noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif'
    save_path = "/data/pt_02718/tmp_data/blink_corrected_eeg/" + subject_id + "/"
    fname_save = f'sr{sfreq}_{cond_name}.fif'
    os.makedirs(save_path, exist_ok=True)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs_noica = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-0.1, tmax=0.07,
                              baseline=tuple([-0.1, -0.01]), preload=True, reject_by_annotation=True)

    # Set montage
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")

    raw_filtered = raw.copy().filter(l_freq=0.5, h_freq=45)
    raw_filtered.resample(1000)
    ica = mne.preprocessing.ICA(n_components=30, max_iter=100, random_state=97)
    ica.fit(raw_filtered)
    ica.exclude = []
    eog_indices, eog_scores = ica.find_bads_eog(raw_filtered, ch_name='FPz')
    ica.exclude = eog_indices
    # barplot of ICA component "EOG match" scores
    ica.plot_scores(eog_scores)
    # plot diagnostics
    ica.plot_properties(raw_filtered, picks=eog_indices)
    # plot ICs applied to raw data, with EOG matches highlighted
    ica.plot_sources(raw_filtered, show_scrollbars=False)
    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    eog_evoked = mne.preprocessing.create_eog_epochs(raw_filtered, ch_name='FPz').average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    eog_evoked.plot_joint()
    ica.plot_sources(eog_evoked)

    ica.apply(raw)  # acts in place

    epochs_ica = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-0.1, tmax=0.07,
                              baseline=tuple([-0.1, -0.01]), preload=True, reject_by_annotation=True)

    fig, ax = plt.subplots()
    plt.plot(epochs_noica.times, epochs_noica.pick(channel).average().get_data().reshape(-1), label='no-ica')
    plt.plot(epochs_ica.times, epochs_ica.pick(channel).average().get_data().reshape(-1), label='ica')
    plt.legend()
    plt.show()

    raw.save(os.path.join(save_path, fname_save), fmt='double', overwrite=True)
    exit()