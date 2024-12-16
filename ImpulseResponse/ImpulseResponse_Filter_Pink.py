# Looking at impulse response of filter
# Taking filter from freq_banded step
# Look at impulse response
# Add noise to signal & look at impulse response

import mne
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from numpy.fft import fft, fftfreq
from scipy import signal
from mne.fixes import minimum_phase
from mne.time_frequency.tfr import morlet
from mne.viz import plot_filter, plot_ideal_filter
import colorednoise as cn


if __name__ == '__main__':
    iir_params = {'order': 2, 'ftype': 'butter'}
    sfreq = 10000

    all_trials_impulse = []
    all_trials_signal = []
    all_trials_signal_unfiltered = []
    all_trials_impulse_unfiltered = []

    for n_trial in np.arange(0, 2000):
        impulse = scipy.signal.unit_impulse(shape=500, idx='mid')
        # Set seed for reproducibility
        noise = cn.powerlaw_psd_gaussian(exponent=1, size=500, random_state=np.random.seed(n_trial))
        signal = impulse + (noise*14)

        # # optionally plot the Power Spectral Density with Matplotlib
        # from matplotlib import mlab
        # from matplotlib import pylab as plt
        # s, f = mlab.psd(noise)
        # plt.loglog(f,s)
        # plt.grid(True)
        # plt.show()
        # exit()

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
    print(f"STD Before Filtering: {std_prefilter}")
    print(f"STD After Filtering: {std_postfilter}")
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