# Looking at impulse response of filter
# Taking filter from freq_banded step
# Look at impulse response
# Add gaussian noise to signal & look at impulse response


import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from numpy.fft import fft, fftfreq
from scipy import signal
from mne.fixes import minimum_phase
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter


if __name__ == '__main__':
    iir_params = {'order': 2, 'ftype': 'butter'}
    sfreq = 10000

    all_trials_impulse = []
    all_trials_signal = []
    all_trials_signal_unfiltered = []
    all_trials_impulse_unfiltered = []
    # impulse = scipy.signal.unit_impulse(shape=500, idx='mid')
    # noise = np.random.normal(0, 10, 500)
    # signal = impulse + noise
    #
    # # Actually filter the impulse versus the signal after filtering
    # filtered_impulse = mne.filter.filter_data(impulse, l_freq=400, h_freq=800, sfreq=sfreq, method='iir',
    #                                           iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')
    # filtered_signal = mne.filter.filter_data(signal, l_freq=400, h_freq=800, sfreq=sfreq, method='iir',
    #                                          iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')

    # # Impulse here is only for sanity checks on length of filter etc.
    # filt = mne.filter.create_filter(
    #     impulse, sfreq, l_freq=400, h_freq=800, method="iir", iir_params=iir_params, verbose=True,
    #     phase='zero'
    # )
    # plot_filter(filt, sfreq, freq=None, gain=None, title="Butterworth order=5", compensate=True)
    # plt.show()
    # exit()

    for n_trial in np.arange(0, 2000):
        impulse = scipy.signal.unit_impulse(shape=500, idx='mid')
        # Set seed for reproducibility
        np.random.seed(n_trial)
        noise = np.random.normal(0, 10.9, 500)
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

    std_prefilter = np.std(all_trials_signal_unfiltered)
    std_postfilter = np.std(all_trials_signal)
    print(std_prefilter)
    print(std_postfilter)
    fig, ax = plt.subplots(2, 2)
    ax[0, 0].plot(np.mean(all_trials_impulse_unfiltered, axis=0))
    ax[0, 0].set_title('Impulse, n_trials=2000')
    ax[0, 1].plot(np.mean(all_trials_signal_unfiltered, axis=0))
    ax[0, 1].set_title('Signal, n_trials=2000')
    ax[1, 0].plot(np.mean(all_trials_impulse, axis=0))
    ax[1, 0].set_title('Filtered Impulse, n_trials=2000')
    ax[1, 1].plot(np.mean(all_trials_signal, axis=0))
    ax[1, 1].set_title('Filtered Signal, n_trials=2000')
    plt.show()
    # Our bandpass filter in original code
    # raw.filter(l_freq=band_dict[band_name][0], h_freq=band_dict[band_name][1], n_jobs=len(raw.ch_names), method='iir',
    #            iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')