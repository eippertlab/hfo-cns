# Get the frequency of the HFO

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from Common_Functions.invert import invert


# Want this to return the frequency of that subjects evoked HFO after CCA
def frequency_of_evoked(srmr_nr, subject, cond_name, freq_band, data_type):
    if srmr_nr == 1:
        p = 'tmp_data'
    elif srmr_nr == 2:
        p = 'tmp_data_2'
    # Cortical Excel files
    xls = pd.ExcelFile(f'/data/pt_02718/{p}/Visibility_Updated.xlsx')
    df_vis_cortical = pd.read_excel(xls, 'CCA_Brain')
    df_vis_cortical.set_index('Subject', inplace=True)

    xls = pd.ExcelFile(f'/data/pt_02718/{p}/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    xls = pd.ExcelFile(f'/data/pt_02718/{p}/Cortical_Timing.xlsx')
    df_timing = pd.read_excel(xls, 'Timing')
    df_timing.set_index('Subject', inplace=True)

    # Spinal Excel files
    xls = pd.ExcelFile(f'/data/pt_02718/{p}/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    xls = pd.ExcelFile(f'/data/pt_02718/{p}/Visibility_Updated.xlsx')
    df_vis_spinal = pd.read_excel(xls, 'CCA_Spinal')
    df_vis_spinal.set_index('Subject', inplace=True)

    subject_id = f'sub-{str(subject).zfill(3)}'
    if data_type == 'eeg':
        fname = f"{freq_band}_{cond_name}.fif"
        input_path = f"/data/pt_02718/{p}/cca_eeg/{subject_id}/"
        df = df_cortical
        df_vis = df_vis_cortical
    elif data_type == 'esg':
        fname = f"{freq_band}_{cond_name}.fif"
        input_path = f"/data/pt_02718/{p}/cca/{subject_id}/"
        df = df_spinal
        df_vis = df_vis_spinal

    epochs = mne.read_epochs(input_path + fname, preload=True)

    # Need to pick channel based on excel sheet
    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
    if channel_no == 0:
        frqY = np.nan
    else:
        channel = f'Cor{channel_no}'
        inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
        epochs = epochs.pick_channels([channel])
        if inv == 'T':
            epochs.apply_function(invert, picks=channel)
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
    study_no = 2

    if study_no == 1:
        subjects = np.arange(1, 37)
        band = 'sigma'
        conditions = ['median', 'tibial']
        data_types = ['esg', 'eeg']
    elif study_no == 2:
        subjects = np.arange(1, 25)
        band = 'sigma'
        conditions = ['med_mixed', 'tib_mixed']
        data_types = ['esg', 'eeg']

    frequencies = [[] for _ in range(0, len(data_types)+len(conditions))]
    # test = [[] for _ in range(0, len(data_types)+len(conditions))]
    count = 0
    for data_t in data_types:
        for condition in conditions:
            for subj in subjects:
                # frequency_of_evoked(subj, condition, band, data_t)
                frequencies[count].append(frequency_of_evoked(study_no, subj, condition, band, data_t))
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
