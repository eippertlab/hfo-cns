import numpy as np
def envelope_noise_reduction(data, noise_data):
    noise_mean = noise_data.mean()

    # data_before = data[0:indices[0]]
    # data_during = data[indices[0]:indices[1]]
    # data_after = data[indices[1]:]
    #
    # cleaned_before = data_before - noise_mean
    # cleaned_after = data_after - noise_mean
    #
    # cleaner_data = np.concatenate((cleaned_before, data_during, cleaned_after))
    cleaner_data = data - noise_mean

    return cleaner_data