# Function which allows the application of CCA weights to broadband epochs data

import numpy as np
def apply_cca_weights(epochs_data, **kwargs):
    # Check all necessary arguments sent in
    required_kws = ["weights"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into apply_cca_weights."

    # Extract all kwargs - more elegant ways to do this
    weights = kwargs['weights']

    # Apply the CCA weights
    st_matrix_long = np.swapaxes(epochs_data, 1, 2).reshape(-1, epochs_data.shape[1]).T
    CCA_concat = st_matrix_long.T @ weights[:, 0:np.shape(weights)[1]]
    CCA_concat = CCA_concat.T

    no_times_long = np.shape(epochs_data)[2]
    no_epochs = np.shape(epochs_data)[0]

    CCA_comps = np.reshape(CCA_concat, (np.shape(weights)[1], no_times_long, no_epochs), order='F')
    CCA_comps = np.swapaxes(CCA_comps, 0, 2)
    CCA_comps = np.swapaxes(CCA_comps, 1, 2)

    if np.shape(CCA_comps)[1] != np.shape(epochs_data)[1]:
        dummy_vals_to_add = np.zeros((no_epochs, np.shape(epochs_data)[1]-np.shape(CCA_comps)[1], no_times_long))
        CCA_comps = np.concatenate((CCA_comps, dummy_vals_to_add), axis=1)

    return CCA_comps
