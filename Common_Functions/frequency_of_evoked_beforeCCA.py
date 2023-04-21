# Get the frequency of the HFO before any CCA manipulation

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from Common_Functions.invert import invert
from Common_Functions.get_conditioninfo import get_conditioninfo


# Want this to return the frequency of that subjects evoked HFO after CCA
def frequency_of_evoked(subject, cond_name, freq_band, data_type):
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Cortical_Timing.xlsx')
    df_timing = pd.read_excel(xls, 'Timing')
    df_timing.set_index('Subject', inplace=True)

    # Set variables
    subject_id = f'sub-{str(subject).zfill(3)}'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    if data_type == 'eeg':
        input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
        fname = f"{freq_band}_{cond_name}.fif"
        if cond_name == 'median':
            trigger_name = 'Median - Stimulation'
            channel = 'CP4'
        elif cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'
            channel = 'Cz'
    elif data_type == 'esg':
        input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
        # input_path_good = "/data/pt_02718/tmp_data/good_trials_spinal/" + subject_id + "/"
        fname = f"{freq_band}_{cond_name}.fif"
        if cond_name == 'median':
            trigger_name = 'Median - Stimulation'
            channel = 'SC6'
        elif cond_name == 'tibial':
            trigger_name = 'Tibial - Stimulation'
            channel = 'L1'

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                        baseline=tuple(iv_baseline), preload=True)

    # Need to pick channel based on excel sheet
    epochs = epochs.pick_channels([channel])
    evoked = epochs.copy().average()
    # evoked = epochs.copy().average().crop(tmin=0.00, tmax=0.07)
    data = evoked.data.reshape(-1)

    SAMPLE_RATE = 5000
    DURATION = evoked.tmax
    # N = int(SAMPLE_RATE * DURATION)
    N = len(data)
    yf = rfft(data)
    xf = rfftfreq(N, 1 / SAMPLE_RATE)

    mY = np.abs(yf)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = xf[locY]  # Get the actual frequency value

    # plt.stem(xf, np.abs(yf))
    # plt.xlim([350, 900])
    # plt.title(f"{subject_id}, {data_type}, {cond_name}")
    # plt.show()

    return frqY


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    band = 'sigma'
    conditions = ['median', 'tibial']
    data_types = ['esg', 'eeg']

    frequencies = [[] for _ in range(0, len(data_types)+len(conditions))]
    # test = [[] for _ in range(0, len(data_types)+len(conditions))]
    count = 0
    for data_t in data_types:
        for condition in conditions:
            for subj in subjects:
                # frequency_of_evoked(subj, condition, band, data_t)
                frequencies[count].append(frequency_of_evoked(subj, condition, band, data_t))
                # test[count].append(count)
            count += 1

    # print(frequencies[0])
    # print(frequencies[1])
    # print(frequencies[2])
    # print(frequencies[3])
    # print(np.shape(frequencies))
    col_names = ['esg_median', 'esg_tibial', 'eeg_median', 'eeg_tibial']
    df = pd.DataFrame({'Subjects': subjects,
                       col_names[0]: frequencies[0],
                       col_names[1]: frequencies[1],
                       col_names[2]: frequencies[2],
                       col_names[3]: frequencies[3]
                       })
    df.set_index('Subjects')
    print(df.describe())
    print(df)
