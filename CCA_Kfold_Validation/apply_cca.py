# Script to actually apply CCA to data, and return the applied data

import numpy as np
from meet import spatfilt

def apply_cca(epochs_train, epochs_apply, epo_cca_train, epo_cca_apply, chans, window):
    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca_train.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs_train.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca_train.get_data(picks=chans)
    epo_cca_data_apply = epo_cca_apply.get_data(picks=chans)
    epo_data_apply = epochs_apply.get_data(picks=chans)

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    no_times = int(window[1] - window[0])
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_apply = np.swapaxes(epo_cca_data_apply, 1, 2).reshape(-1, epo_cca_data_apply.shape[1]).T
    st_matrix_long_apply = np.swapaxes(epo_data_apply, 1, 2).reshape(-1, epo_data_apply.shape[1]).T

    # Run CCA to get the filters
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)  # avg and st from the train data

    all_components = len(r)

    # Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
    CCA_concat = st_matrix_long_apply.T @ W_st[:, 0:all_components]
    CCA_concat = CCA_concat.T

    # Spatial Patterns
    A_st = np.cov(st_matrix_apply) @ W_st

    # Reshape
    no_times_long = np.shape(epochs_apply.get_data())[2]
    no_epochs = np.shape(epochs_apply.get_data())[0]

    # Perform reshape
    CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)
    selected_components = all_components  # Just keeping all for now to avoid rerunning

    return CCA_comps, A_st, W_st, r, all_components
