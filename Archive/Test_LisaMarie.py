import mne
import numpy as np
from meet import spatfilt
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':

    plot_graphs = True

    epochs_all = mne.read_epochs('/data/hu_bailey/Downloads/es04_preprocessed-epo_FIR.fif', preload=True)
    # Epochs run from -2.0s to 4.0s
    epochs_all.set_eeg_reference(['Fz'])
    # epochs_all.set_eeg_reference(ref_channels='average')
    epochs_all.filter(l_freq=None, h_freq=30)
    annotations = epochs_all.get_annotations_per_epoch()

    idx_rep_exp = [idx for idx, item in enumerate(annotations) if '2nd/repetition/expected' in item[1]]
    idx_rep_une = [idx for idx, item in enumerate(annotations) if '2nd/repetition/unexpected' in item[1]]
    idx_omi_exp = [idx for idx, item in enumerate(annotations) if '2nd/omission/expected' in item[1]]
    idx_omi_une = [idx for idx, item in enumerate(annotations) if '2nd/omission/unexpected' in item[1]]

    epo_exp = epochs_all[idx_rep_exp]
    epo_une = epochs_all[idx_rep_une]
    # epo_omi_exp_sub = epochs_all[idx_omi_exp]
    # epo_omi_une_sub = epochs_all[idx_omi_une]

    # window_times = [198 / 1000, 702 / 1000]
    window_times = [198 / 1000, 302 / 1000]

    # Crop the epochs
    window = epochs_all.time_as_index(window_times)
    epo_cca = epochs_all.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)

    # Prepare matrices for cca
    ##### Average matrix
    epo_av = epo_cca.copy().average().data.T
    # Now want channels x observations matrix #np.shape()[0] gets number of trials
    # Epo av is no_times x no_channels (10x40)
    # Want to repeat this to form an array thats no. observations x no.channels (20000x40)
    # Need to repeat the array, no_trials/times amount along the y axis
    avg_matrix = np.tile(epo_av, (int((np.shape(epochs_all.get_data())[0])), 1))
    avg_matrix = avg_matrix.T  # Need to transpose for correct form for function - channels x observations

    ##### Single trial matrix
    epo_cca_data = epo_cca.get_data()
    epo_data = epochs_all.get_data()
    epo_data_exp = epo_exp.get_data()
    epo_data_une = epo_une.get_data()

    # 0 to access number of epochs, 1 to access number of channels
    # channels x observations
    no_times = int(window[1] - window[0])
    # Need to transpose to get it in the form CCA wants
    st_matrix = np.swapaxes(epo_cca_data, 1, 2).reshape(-1, epo_cca_data.shape[1]).T
    st_matrix_long = np.swapaxes(epo_data, 1, 2).reshape(-1, epo_data.shape[1]).T
    st_matrix_long_exp = np.swapaxes(epo_data_exp, 1, 2).reshape(-1, epo_data_exp.shape[1]).T
    st_matrix_long_une = np.swapaxes(epo_data_une, 1, 2).reshape(-1, epo_data_une.shape[1]).T

    # Run CCA
    W_avg, W_st, r = spatfilt.CCA_data(avg_matrix, st_matrix)

    all_components = len(r)

    for epoch_type in ['rep_expected', 'rep_unexpected']:
        if epoch_type == 'rep_expected':
            epochs = epochs_all[idx_rep_exp]
            st_matrix_long = st_matrix_long_exp
        elif epoch_type == 'rep_unexpected':
            st_matrix_long = st_matrix_long_une
            epochs = epochs_all[idx_rep_une]
        else:
            print('Invalid epoch type requested')
            break

        # Apply obtained weights to the long dataset (dimensions 40x9) - matrix multiplication
        CCA_concat = st_matrix_long.T @ W_st[:, 0:all_components]
        CCA_concat = CCA_concat.T

        # Spatial Patterns
        A_st = np.cov(st_matrix) @ W_st

        # Reshape - (900, 2000, 9)
        no_times_long = np.shape(epochs.get_data())[2]
        no_epochs = np.shape(epochs.get_data())[0]

        # Perform reshape
        CCA_comps = np.reshape(CCA_concat, (all_components, no_times_long, no_epochs), order='F')

        # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
        CCA_comps = np.swapaxes(CCA_comps, 0, 2)
        CCA_comps = np.swapaxes(CCA_comps, 1, 2)
        selected_components = all_components  # Just keeping all for now to avoid rerunning

        #######################  Epoch data class to store the information ####################
        data = CCA_comps[:, 0:selected_components, :]
        events = epochs.events
        event_id = epochs.event_id
        tmin = -2.0
        sfreq = 500

        ch_names = []
        ch_types = []
        for i in np.arange(0, all_components):
            ch_names.append(f'Cor{i + 1}')
            ch_types.append('eeg')

        # Initialize an info structure
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=ch_types,
            sfreq=sfreq
        )

        # Create and save
        cca_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
        cca_epochs = cca_epochs.apply_baseline(baseline=(None, 0))
        # cca_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

        ################################ Plotting Graphs #######################################
        if plot_graphs:
            if epoch_type == 'rep_expected':
                fig_test, ax_test = plt.subplots()
            ############ Time Course of First 4 components ###############
            # cca_epochs and cca_epochs_d both already baseline corrected before this point
            fig = plt.figure()
            for icomp in np.arange(0, 4):
                plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}, r={r[icomp]:.3f}')
                # Want to plot Cor1 - Cor4
                # Plot for the mixed nerve data
                # get_data returns (n_epochs, n_channels, n_times)
                data = cca_epochs.get_data(picks=[f'Cor{icomp + 1}'])
                to_plot = np.mean(data[:, 0, :], axis=0)
                plt.plot(cca_epochs.times, to_plot)
                if icomp == 0:
                    ax_test.plot(cca_epochs.times, to_plot, label=epoch_type)
                plt.axvline(0.18, color='r', linewidth=0.3)
                plt.axvline(0.27, color='r', linewidth=0.3)
                plt.axvline(0.7, color='r', linewidth=0.3)
                plt.xlim([-0.1, 2.0])
                plt.xlabel('Time [s]')
                plt.ylabel('Amplitude [A.U.]')
                plt.legend()
                # plt.tight_layout()
            plt.suptitle(epoch_type)
            plt.tight_layout()
            # plt.close(fig)

            ######################## Plot image for cca_epochs ############################
            # cca_epochs and cca_epochs_d both already baseline corrected before this point
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
            axes = [ax1, ax2, ax3, ax4]
            cropped = cca_epochs.copy().crop(tmin=-0.1, tmax=2.0)
            cmap = mpl.colors.ListedColormap(["blue", "green", "red"])

            vmin = -2
            vmax = 2

            for icomp in np.arange(0, 4):
                cropped.plot_image(picks=f'Cor{icomp + 1}', combine=None, cmap='jet', evoked=False, show=False,
                                   axes=axes[icomp], title=f'Component {icomp + 1}', colorbar=False, group_by=None,
                                   vmin=vmin, vmax=vmax, units=dict(eeg='V'), scalings=dict(eeg=1))

            plt.suptitle(epoch_type)
            plt.tight_layout()
            fig.subplots_adjust(right=0.85)
            ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            # mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm, spacing='proportional')
            mpl.colorbar.ColorbarBase(ax5, cmap='jet', norm=norm)
            # plt.close(fig)
    ax_test.legend()
    plt.show()
