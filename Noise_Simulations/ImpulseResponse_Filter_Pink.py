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
    scaler = 5.1
    save_path = f"/data/pt_02718/tmp_data/noise_simulations_{scaler}timesnoise/"
    os.makedirs(save_path, exist_ok=True)

    for n_iterations in np.arange(0, 60):
        all_trials_impulse = []
        all_trials_signal = []
        all_trials_signal_unfiltered = []
        all_trials_impulse_unfiltered = []

        for n_trial in np.arange(0, 2000):
            impulse = scipy.signal.unit_impulse(shape=2000, idx='mid')
            # Set seed for reproducibility
            noise = cn.powerlaw_psd_gaussian(exponent=1, size=2000, random_state=np.random.seed(n_trial))
            signal = impulse + (noise*scaler)

            all_trials_signal_unfiltered.append(signal)
            all_trials_impulse_unfiltered.append(impulse)

            # Actually filter the impulse versus the signal after filtering
            filtered_impulse = mne.filter.filter_data(impulse, l_freq=400, h_freq=800, sfreq=sfreq, method='iir',
                                                      iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')
            filtered_signal = mne.filter.filter_data(signal, l_freq=400, h_freq=800, sfreq=sfreq, method='iir',
                                                     iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')

            all_trials_impulse.append(filtered_impulse)
            all_trials_signal.append(filtered_signal)

        noise_period_unfiltered = [x[10:461] for x in all_trials_signal_unfiltered]
        noise_period_filtered = [x[10:461] for x in all_trials_signal]
        np.save(f"{save_path}{n_iterations}_unfiltered", all_trials_signal_unfiltered)
        np.save(f"{save_path}{n_iterations}_filtered", all_trials_signal)
        std_prefilter = np.std(noise_period_unfiltered)
        std_postfilter = np.std(noise_period_filtered)
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
        plt.tight_layout()
        plt.savefig(f"{save_path}{n_iterations}_iteration")
        plt.close()
    # Our bandpass filter in original code
    # raw.filter(l_freq=band_dict[band_name][0], h_freq=band_dict[band_name][1], n_jobs=len(raw.ch_names), method='iir',
    #            iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')