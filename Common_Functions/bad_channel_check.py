#################################################################################################
# Generate plots to identify bad channels to mark for removal
#################################################################################################

import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Common_Functions.get_conditioninfo import get_conditioninfo


def bad_channel_check(subject, condition, srmr_nr, sampling_rate, channel_type):
    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    # Select the right files based on the data_string
    if channel_type == 'esg':
        if srmr_nr == 1:
            input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
            fname = f'ssp6_cleaned_{cond_name}.fif'
            figure_path = "/data/pt_02718/tmp_data/bad_channels_esg/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)
        elif srmr_nr == 2:
            input_path = "/data/pt_02718/tmp_data_2/ssp_cleaned/" + subject_id + "/"
            fname = f'ssp6_cleaned_{cond_name}.fif'
            figure_path = "/data/pt_02718/tmp_data_2/bad_channels_esg/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

    elif channel_type == 'eeg':
        if srmr_nr == 1:
            input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
            fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif'
            figure_path = "/data/pt_02718/tmp_data/bad_channels_eeg/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)
        elif srmr_nr == 2:
            input_path = "/data/pt_02718/tmp_data_2/imported/" + subject_id + "/"
            fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif'
            figure_path = "/data/pt_02718/tmp_data_2/bad_channels_eeg/" + subject_id + "/"
            os.makedirs(figure_path, exist_ok=True)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_long_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_long_end', 'var_value'].iloc[0]]

    ##########################################################################################
    # Generates psd - can click on plot to find bad channel name
    ##########################################################################################
    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Power spectral density of {channel_type} data sub-{subject}")
    fig.tight_layout(pad=3.0)
    if 'TH6' in raw.ch_names:  # Can't use zero value in spectrum for channel
        raw.copy().drop_channels('TH6').compute_psd(fmax=2000).plot(axes=axes, show=False)
    else:
        raw.compute_psd(fmax=2000).plot(axes=axes, show=False)
    axes.set_ylim([-80, 50])
    plt.savefig(figure_path + f'psd_{cond_name}.png')

    ###########################################################################################
    # Squared log means of each channel
    ###########################################################################################
    events, event_ids = mne.events_from_annotations(raw)
    if srmr_nr == 1:
        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    elif srmr_nr == 2:   # Because we have e.g. med1, med2 and med12
        event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline), preload=True)
    fig, axes = plt.subplots(figsize=(12, 8))
    fig.suptitle(f"Squared log means per {channel_type} epoch sub-{subject}")
    table = epochs.to_data_frame()
    table = table.drop(columns=["time", "ECG", "TH6"])
    table = pd.concat([table.iloc[:, :2], np.square(table.iloc[:, 2:])], axis=1)
    table = pd.concat([table.iloc[:, :2], np.log(table.iloc[:, 2:])], axis=1)
    means = table.groupby(['epoch']).mean().T  # average
    ax_i = axes.matshow(means, aspect='auto')  # plots mean values by colorscale
    plt.colorbar(ax_i, ax=axes)
    axes.set_yticks(np.arange(0, len(list(means.index))), list(means.index))  # Don't hardcode 41
    axes.tick_params(labelbottom=True)
    plt.savefig(figure_path + f'meanlog_{cond_name}.png')
    plt.show()

    bad_chans = list(map(str, input("Enter bad channels (separated by a space, press enter if none): ").split()))
    filename = figure_path + f'bad_channels_{cond_name}.txt'
    with open(filename, mode="w") as outfile:
        for s in bad_chans:
            outfile.write("%s\n" % s)

    if bad_chans:  # If there's something in here append it to the bads
        raw.info['bads'].extend(bad_chans)

        # Save them with the bad channels now marked
        if channel_type == 'esg':
            raw.save(f"{input_path}{fname}", fmt='double', overwrite=True)

        elif channel_type == 'eeg':
            raw.save(f"{input_path}{fname}", fmt='double', overwrite=True)
