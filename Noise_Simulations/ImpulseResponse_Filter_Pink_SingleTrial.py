# Looking at impulse response of filter
# Taking filter from freq_banded step
# Look at impulse response
# Add noise to signal & look at impulse response
import os

import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import colorednoise as cn


if __name__ == '__main__':
    sfreq = 10000
    # Just to have an info structure to use mne methods
    input_path = f"/data/pt_02718/tmp_data/imported/sub-001/"
    fname = f"noStimart_sr5000_median_withqrs_eeg.fif"
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    raw.pick('CP4')

    all_trials_impulse = []
    all_trials_signal = []
    all_trials_signal_unfiltered = []
    all_trials_impulse_unfiltered = []
    for n_trial in np.arange(0, 2000):
        impulse = scipy.signal.unit_impulse(shape=2000, idx='mid')
        # Set seed for reproducibility
        noise = cn.powerlaw_psd_gaussian(exponent=1, size=2000, random_state=np.random.seed(n_trial))
        signal = impulse + noise

        all_trials_signal_unfiltered.append(signal)
        all_trials_impulse_unfiltered.append(impulse)

        # Actually filter the impulse versus the signal after filtering
        filtered_impulse = mne.filter.filter_data(impulse, l_freq=400, h_freq=800, sfreq=sfreq, method='iir',
                                                  iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')
        filtered_signal = mne.filter.filter_data(signal, l_freq=400, h_freq=800, sfreq=sfreq, method='iir',
                                                 iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')

        all_trials_impulse.append(filtered_impulse)
        all_trials_signal.append(filtered_signal)

    # All have shape, n_trials, n_samples
    for_tfr_impulse = np.expand_dims(all_trials_impulse, 1)
    for_tfr_signal = np.expand_dims(all_trials_signal, 1)
    st_power_impulse, itc_impulse, freqs_impulse = mne.time_frequency.tfr_array_stockwell(for_tfr_impulse, sfreq=sfreq, fmin=0, fmax=1200)
    st_power_signal, itc_signal, freqs_signal = mne.time_frequency.tfr_array_stockwell(for_tfr_signal, sfreq=sfreq, fmin=0, fmax=1200)
    tfr_impulse = mne.time_frequency.AverageTFRArray(raw.info, st_power_impulse, np.arange(0, 2000), freqs_impulse)
    tfr_signal = mne.time_frequency.AverageTFRArray(raw.info, st_power_signal, np.arange(0, 2000), freqs_signal)
    print(np.shape(st_power_signal))
    noise_period_unfiltered = [x[10:461] for x in all_trials_signal_unfiltered]
    noise_period_filtered = [x[10:461] for x in all_trials_signal]
    std_prefilter = np.std(noise_period_unfiltered)
    std_postfilter = np.std(noise_period_filtered)
    fig, ax = plt.subplots(3, 2)
    ax[0, 0].plot(np.mean(all_trials_impulse_unfiltered, axis=0))
    ax[0, 0].set_title('Impulse, n_trials=2000')
    ax[0, 1].plot(np.mean(all_trials_signal_unfiltered, axis=0))
    ax[0, 1].set_title('Signal, n_trials=2000')
    ax[1, 0].plot(np.mean(all_trials_impulse, axis=0))
    ax[1, 0].set_title('Filtered Impulse, n_trials=2000')
    ax[1, 1].plot(np.mean(all_trials_signal, axis=0))
    ax[1, 1].set_title('Filtered Signal, n_trials=2000')
    vmin = -150
    vmax = 150
    # ToDo: Will only plot one or the other
    # tfr_impulse.plot([0], tmin=0, tmax=1999, fmin=0, fmax=1200, dB=True, axes=ax[2, 0], vmin=vmin, vmax=vmax,
    #                  combine='mean', baseline=tuple([10, 451]), mode='ratio')
    tfr_signal.plot([0], tmin=0, tmax=1999, fmin=0, fmax=1200, dB=True, axes=ax[2, 1], vmin=vmin, vmax=vmax,
                    combine='mean', baseline=tuple([10, 451]), mode='ratio')
    plt.tight_layout()
    plt.show()
    # Our bandpass filter in original code
    # raw.filter(l_freq=band_dict[band_name][0], h_freq=band_dict[band_name][1], n_jobs=len(raw.ch_names), method='iir',
    #            iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')