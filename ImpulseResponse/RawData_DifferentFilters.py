# Look at the data before filtering to see how close we are to an impulse response


import mne
import matplotlib.pyplot as plt
import pandas as pd
from Common_Functions.evoked_from_raw import evoked_from_raw

if __name__ == '__main__':
    # input_file = "/data/pt_02718/tmp_data/imported/sub-006/noStimart_sr5000_median_withqrs_eeg.fif"
    input_file = "/data/pt_02718/tmp_data/ssp_cleaned/sub-006/ssp6_cleaned_median.fif"
    raw = mne.io.read_raw_fif(input_file, preload=True)
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    for order in [1, 2, 3, 4, 5, 6]:
        raw_filt = raw.copy().filter(l_freq=400, h_freq=800, n_jobs=len(raw.ch_names), method='iir',
                                     iir_params={'order': order, 'ftype': 'butter'}, phase='zero')
        evoked = evoked_from_raw(raw_filt, iv_epoch, iv_baseline, 'Median - Stimulation', False)
        plt.figure()
        plt.plot(evoked.times, evoked.pick(['SC6']).get_data().reshape(-1))
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (V)')
        plt.xlim([-0.05, 0.065])
        plt.axvline(0.013, color='red', linewidth=0.5, label='13ms')
        plt.title(f"Filter Order: {order}")
        plt.legend()
        plt.show()
        plt.close()
