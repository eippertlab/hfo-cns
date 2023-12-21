# Exponential fit and subtract from segments of data

import mne
import numpy as np
import matplotlib.pyplot as plt


def subtract_fit(data, **kwargs):
    # Data is a single channels time data
    # Check all necessary arguments sent in
    required_kws = ["times", "indices"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into exponential fit."

    # Extract all kwargs - more elegant ways to do this
    times = kwargs['times']
    indices = kwargs['indices']
    all_time = kwargs['all_time']

    p = np.polyfit(times, np.log(data[indices[0]:indices[1]]), 1)
    a = np.exp(p[1])
    b = p[0]
    x_fitted = times
    y_fitted = a * np.exp(b * x_fitted)

    ax = plt.axes()
    # ax.scatter(data, label='Raw data')
    ax.plot(all_time, data)
    ax.scatter(times, data[indices[0]:indices[1]], label='Raw data')
    ax.plot(x_fitted, y_fitted, 'k', label='Fitted curve')
    ax.set_title('Using polyfit() to fit an exponential function')
    ax.set_ylabel('y-Values')
    ax.set_xlabel('x-Values')
    ax.legend()

    plt.show()

    exit()

    # # Need to subtract exp and return data
    # data[indices[0]:indices[1]] = data[indices[0]:indices[1]] - y_fitted
    #
    # return data

