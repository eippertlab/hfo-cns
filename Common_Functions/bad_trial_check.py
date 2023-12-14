#######################################################################################
# Annotate bad segments of data based on amplitude thresholds defined as mean +- 3*std
#######################################################################################

import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from mne import Annotations
from Common_Functions.get_channels import get_channels


def bad_trial_check(subject, condition, srmr_nr, sampling_rate, channel_type):
    subject_id = f'sub-{str(subject).zfill(3)}'
    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name

    if channel_type == 'esg':
        if srmr_nr == 1:
            input_path = "/data/pt_02718/tmp_data/ssp_cleaned/" + subject_id + "/"
            fname = f'ssp6_cleaned_{cond_name}.fif'
        else:
            input_path = "/data/pt_02718/tmp_data_2/ssp_cleaned/" + subject_id + "/"
            fname = f'ssp6_cleaned_{cond_name}.fif'

    elif channel_type == 'eeg':
        if srmr_nr == 1:
            input_path = "/data/pt_02718/tmp_data/imported/" + subject_id + "/"
            fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif'
        elif srmr_nr == 2:
            input_path = "/data/pt_02718/tmp_data_2/imported/" + subject_id + "/"
            fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_eeg.fif'

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Computes the rejection dictionary automatically (global)
    eeg_chans, esg_chans, bipolar_chans = get_channels(subject_nr=subject, includesEcg=False, includesEog=False,
                                                       study_nr=srmr_nr)

    # Need to get data in frequency zone of interest to compute mean and standard deviation
    # (n_channels, n_times)
    if channel_type == 'esg':
        data = raw.copy().filter(l_freq=400, h_freq=1400, n_jobs=len(raw.ch_names), method='iir',
                          iir_params={'order': 2, 'ftype': 'butter'}, phase='zero').get_data(picks=esg_chans)
    elif channel_type == 'eeg':
        data = raw.copy().filter(l_freq=400, h_freq=1400, n_jobs=len(raw.ch_names), method='iir',
                          iir_params={'order': 2, 'ftype': 'butter'}, phase='zero').get_data(picks=eeg_chans)

    meanAllChan = np.mean(data, axis=0)
    stdAllChan = np.std(data, axis=0)
    pos_3SDChan = meanAllChan + 3 * stdAllChan
    neg_3SDChan = meanAllChan - 3 * stdAllChan
    amplitudeThreshold = max(np.max(np.vstack((pos_3SDChan, abs(neg_3SDChan))), axis=0))

    minAllChan = abs(np.min(data, axis=0))
    maxAllChan = abs(np.max(data, axis=0))
    absMinMaxAllChan = np.max(np.vstack((minAllChan, maxAllChan)), axis=0)
    badPoints = np.where(absMinMaxAllChan > amplitudeThreshold, True, False)
    # Returns logical mask - True where condition is true, false where condition is false

    # Get timing of points threshold is exceeded
    sample_indices = np.argwhere(badPoints)
    if sample_indices.size != 0:
        sample_indices = sample_indices.reshape(-1)
        # Add bad amplitude events as annotations - have onset 2.5 ms before bad segment
        bad_amp_events = [x / sampling_rate - 0.0025 for x in sample_indices]  # Divide by sampling rate to make times
        annotations = Annotations(bad_amp_events, duration=0.005, description="BAD_amp")
        # Will be 2.5ms before and 2.5ms after the detected bad amplitude
        raw.set_annotations(raw.annotations + annotations)

        if channel_type == 'esg':
            raw.save(f"{input_path}{fname}", fmt='double', overwrite=True)

        elif channel_type == 'eeg':
            raw.save(f"{input_path}{fname}", fmt='double', overwrite=True)
