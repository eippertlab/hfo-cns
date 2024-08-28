# Compute SNR of the data for each mode (ESG and CNSLevelSpecificFunctions) and each condition (median, tibial, med_mixed, tib_mixed)
# This is for the low frequency SEPs
# The SNR was estimated by dividing the evoked response peak amplitude
# (absolute value) by the standard deviation of the LEP waveform in
# the pre-stimulus interval
# https://www.sciencedirect.com/science/article/abs/pii/S105381190901297X

def calculate_snr_lowfreqcca_polarity(evoked_channel, noise_window, signal_window, sep_latency, data_type, cond_name):
    # evoked is a single channel of evoked data in MNE
    # noise_window is the period in seconds where noise is
    # signal_window is used to determine bounds around sep_latency where signal should be (in seconds)
    # sep_latency is the time in s where that subjects low frequency SEP peaks

    if data_type == 'subcortical':
        ch_name, latency, amplitude = evoked_channel.get_peak(tmin=sep_latency - signal_window,
                                                              tmax=sep_latency + signal_window/2,
                                                              mode='pos', return_amplitude=True)
    elif data_type == 'spinal':
        ch_name, latency, amplitude = evoked_channel.get_peak(tmin=sep_latency - signal_window,
                                                              tmax=sep_latency + signal_window,
                                                              mode='neg', return_amplitude=True)
    else:
        if cond_name in ['median', 'med_mixed']:
            ch_name, latency, amplitude = evoked_channel.get_peak(tmin=sep_latency - signal_window,
                                                                  tmax=sep_latency + signal_window,
                                                                  mode='neg', return_amplitude=True)
        else:
            ch_name, latency, amplitude = evoked_channel.get_peak(tmin=sep_latency - signal_window,
                                                                  tmax=sep_latency + signal_window,
                                                                  mode='pos', return_amplitude=True)

    data = evoked_channel.copy().crop(tmin=noise_window[0], tmax=noise_window[1]).get_data()
    sd = data.std()

    return abs(amplitude / sd)

