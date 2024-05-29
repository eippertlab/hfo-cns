# Function to calculate the SNR of a single channel evoked HFO

def calculate_snr(evoked_channel, noise_window, signal_window, sep_latency, data_type):
    # evoked is a single channel of evoked data in MNE
    # noise_window is the period in seconds where noise is
    # signal_window is used to determine bounds around sep_latency where signal should be (in seconds)
    # sep_latency is expected time of sep peak

    if data_type == 'subcortical':
        # Subcortical time window is now asymmetric
        ch_name, latency, amplitude = evoked_channel.get_peak(tmin=sep_latency - signal_window,
                                                              tmax=sep_latency + signal_window/2,
                                                              mode='abs',
                                                              return_amplitude=True)  # Don't care about sign
    else:
        ch_name, latency, amplitude = evoked_channel.get_peak(tmin=sep_latency-signal_window, tmax=sep_latency+signal_window,
                                                              mode='abs', return_amplitude=True)  # Don't care about sign

    data = evoked_channel.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data()
    sd = data.std()

    return abs(amplitude/sd)
