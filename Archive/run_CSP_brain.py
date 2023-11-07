import mne
from scipy.io import loadmat
import pandas as pd
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import os
import numpy as np
from meet.spatfilt import CSP
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import pickle
import matplotlib.pyplot as plt

# if __name__ == '__main__':


def run_CSP(subject, condition, srmr_nr, freq_band, sfreq):
    # # For testing
    # condition = 3
    # srmr_nr = 1
    # subject = 1
    # freq_band = 'kappa'
    # s_freq = 5000

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    # Select the right files based on the data_string
    input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
    fname = f"{freq_band}_{cond_name}.fif"
    save_path = "/data/pt_02718/tmp_data/csp_eeg/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

    raw = mne.io.read_raw_fif(input_path + fname, preload=True)

    # Set montage
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

    # now create epochs based on the trigger names
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1] - 1 / 1000,
                        baseline=tuple(iv_baseline), preload=True)

    if cond_name == 'median':
        epochs = epochs.pick_channels(eeg_chans, ordered=True)
        if freq_band == 'sigma':
            window_times = [15.4 / 1000, 24.8 / 1000]
        elif freq_band == 'kappa':
            window_times = [18.4 / 1000, 24.8 / 1000]
        sep_latency = 20
    elif cond_name == 'tibial':
        epochs = epochs.pick_channels(eeg_chans, ordered=True)
        if freq_band == 'sigma':
            window_times = [32 / 1000, 44 / 1000]
        elif freq_band == 'kappa':
            window_times = [32 / 1000, 44 / 1000]
        sep_latency = 40
    else:
        print('Invalid condition name attempted for use')
        exit()

    # Crop the epochs
    epo_csp = epochs.copy().crop(tmin=window_times[0], tmax=window_times[1], include_tmax=False)
    epo_contrast = epochs.copy().crop(tmin=80/1000, tmax=90/1000, include_tmax=False)

    # Need to then extract data - variables in rows, observations in columns, same as CCA
    epo_csp_data = epo_csp.get_data()
    epo_contrast_data = epo_contrast.get_data()
    st_matrix_csp = np.swapaxes(epo_csp_data, 1, 2).reshape(-1, epo_csp_data.shape[1]).T
    st_matrix_contrast = np.swapaxes(epo_contrast_data, 1, 2).reshape(-1, epo_contrast_data.shape[1]).T

    # Train filters
    filter, eigval = CSP(st_matrix_csp, st_matrix_contrast)
    # filtered_data = filter.T.dot(st_matrix_csp)

    # Apply filters to the whole epochs
    epoch_data = epochs.get_data()
    st_matrix = np.swapaxes(epoch_data, 1, 2).reshape(-1, epoch_data.shape[1]).T
    epoch_filtered_data = filter.T.dot(st_matrix)

    # Spatial Patterns
    A_st = np.cov(st_matrix) @ filter

    # Reshape
    no_channels = len(epochs.ch_names)
    no_times_long = np.shape(epochs.get_data())[2]
    no_epochs = np.shape(epochs.get_data())[0]
    CSP_filtered = np.reshape(epoch_filtered_data, (no_channels, no_times_long, no_epochs), order='F')

    # print(np.shape(filter))  # (n_channels, n_channels), columns are filters - keep top
    # print(np.shape(CSP_filtered))  # (n_filters, n_times, n_epochs)
    # exit()

    ##########################################################################################
    # Save to epoch structure
    ##########################################################################################
    # Now we have CCA comps, get the data in the axes format MNE likes (n_epochs, n_channels, n_times)
    CSP_comps = np.swapaxes(CSP_filtered, 0, 2)
    CSP_comps = np.swapaxes(CSP_comps, 1, 2)
    selected_channels = len(epochs.ch_names)  # Just keeping all for now to avoid rerunning

    data = CSP_comps[:, 0:selected_channels, :]
    events = epochs.events
    event_id = epochs.event_id
    tmin = iv_epoch[0]
    sfreq = sfreq

    ch_names = []
    ch_types = []
    for i in np.arange(0, selected_channels):
        ch_names.append(f'Comp {i+1}')
        ch_types.append('eeg')

    # Initialize an info structure
    info = mne.create_info(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq
    )

    # Create and save
    csp_epochs = mne.EpochsArray(data, info, events, tmin, event_id)
    csp_epochs = csp_epochs.apply_baseline(baseline=tuple(iv_baseline))
    csp_epochs.save(os.path.join(save_path, fname), fmt='double', overwrite=True)

    ################################ Save Spatial Pattern #################################
    afile = open(save_path + f'A_st_{freq_band}_{cond_name}.pkl', 'wb')
    pickle.dump(A_st, afile)
    afile.close()

    ############################# Top Spatial Patterns ##############################
    figure_path_spatial = f'/data/p_02718/Images/CSP_eeg/ComponentIsopotentialPlots/{subject_id}/'
    os.makedirs(figure_path_spatial, exist_ok=True)
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for icomp in np.arange(0, 4):  # Plot for each of four components
        # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
        chan_labels = epochs.ch_names
        mne.viz.plot_topomap(data=A_st[:, icomp], pos=res, ch_type='eeg', sensors=True, names=None,
                             contours=6, outlines='head', sphere=None, image_interp='cubic',
                             extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                             cnorm=None, axes=axes[icomp], show=False)
        axes[icomp].set_title(f'Component {icomp + 1}')
        divider = make_axes_locatable(axes[icomp])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb = fig.colorbar(axes[icomp].images[-1], cax=cax, shrink=0.6, orientation='vertical')

    plt.savefig(figure_path_spatial + f'{freq_band}_{cond_name}.png')
    plt.close(fig)

    ############################# Time Course Plots ####################################
    figure_path_time = f'/data/p_02718/Images/CSP_eeg/ComponentTimePlots/{subject_id}/'
    os.makedirs(figure_path_time, exist_ok=True)

    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    for icomp in np.arange(0, 4):  # Plot for each of four components
        csp_evoked = csp_epochs.copy().pick_channels([f'Comp {icomp+1}']).crop(0.01, 0.07).average()
        csp_evoked.plot(axes=axes[icomp], show=False, titles=dict(eeg=f'Comp {icomp+1}'))
    plt.tight_layout()
    plt.savefig(figure_path_time + f'{freq_band}_{cond_name}.png')
    plt.close(fig)

    ############################# Single Trial Plots #####################################
    figure_path_st = f'/data/p_02718/Images/CSP_eeg/ComponentSinglePlots/{subject_id}/'
    os.makedirs(figure_path_st, exist_ok=True)
    fig, axes = plt.subplots(2, 2)
    axes = axes.flatten()
    vmin = -1.6e6
    vmax = 2.2e6
    for icomp in np.arange(0, 4):  # Plot for each of four components
        csp_epochs.copy().crop(0.01, 0.07).plot_image(picks=f'Comp {icomp+1}', combine=None, evoked=False, show=False,
                                                      title=f'Component {icomp+1}', colorbar=False, group_by=None,
                                                      vmin=vmin, vmax=vmax, axes=axes[icomp])
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)
    ax5 = fig.add_axes([0.9, 0.1, 0.01, 0.8])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # mpl.colorbar.ColorbarBase(ax5, cmap=cmap, norm=norm, spacing='proportional')
    mpl.colorbar.ColorbarBase(ax5, cmap='RdBu_r', norm=norm)
    plt.savefig(figure_path_st + f'{freq_band}_{cond_name}.png')
    plt.close(fig)

    ############################ Combine to one Image ##########################
    figure_path = f'/data/p_02718/Images/CSP_eeg/ComponentPlots/{subject_id}/'
    os.makedirs(figure_path, exist_ok=True)

    spatial = plt.imread(figure_path_spatial + f'{freq_band}_{cond_name}.png')
    time = plt.imread(figure_path_time + f'{freq_band}_{cond_name}.png')
    single_trial = plt.imread(figure_path_st + f'{freq_band}_{cond_name}.png')

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].imshow(time)
    axes[0, 0].axis('off')
    axes[0, 1].imshow(spatial)
    axes[0, 1].axis('off')
    axes[1, 0].imshow(single_trial)
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    plt.subplots_adjust(top=0.95, wspace=0, hspace=0)

    plt.suptitle(f'Subject {subject}, {freq_band}_{cond_name}')
    plt.savefig(figure_path + f'{freq_band}_{cond_name}.png')
    plt.close(fig)
