# For digit stimulation (fingers & toes)
# Plot single subject spatial pattern


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Common_Functions.check_excel_exist import check_excel_exist
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':

    freq_band = 'sigma'
    srmr_nr = 2

    if srmr_nr != 2:
        print('Error: This script is only designed to work for experiment 2 (digit stimulation)')

    subjects = np.arange(1, 25)  # 1 through 24 to access subject data
    conditions = [2, 4]  # Conditions of interest - med_digits and tib_digits
    component_fname = '/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated_Digits.xlsx'
    visibility_fname = '/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated_Digits.xlsx'
    figure_path = '/data/p_02718/Images_2/CCA_eeg_thalamic_digits/ComponentIsopotentialPlots/'
    os.makedirs(figure_path, exist_ok=True)

    # Get a raw file so I can use the montage
    raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data_2/freq_banded_eeg/sub-001/sigma_med_mixed.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)


    for condition in conditions:
        # Set variables
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_names = cond_info.trigger_name  # Will return list of 3, we just want the 12 one
        trigger_name = trigger_names[2]

        for subject in subjects:
            eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
            subject_id = f'sub-{str(subject).zfill(3)}'
            # Spatial pattern
            fname = f"A_st_{freq_band}_{cond_name}.pkl"
            input_path = "/data/pt_02718/tmp_data_2/cca_eeg_thalamic/" + subject_id + "/"
            with open(f'{input_path}{fname}', 'rb') as f:
                A_st = pickle.load(f)

            ####### Isopotential Plots for the first 4 components ########
            # fig, axes = plt.figure()
            fig, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            for icomp in np.arange(0, 4):  # Plot for each of four components
                # plt.subplot(2, 2, icomp + 1, title=f'Component {icomp + 1}')
                chan_labels = raw.pick_channels(eeg_chans).ch_names
                mne.viz.plot_topomap(data=A_st[:, icomp], pos=res, ch_type='eeg', sensors=True, names=None,
                                     contours=6, outlines='head', sphere=None, image_interp='cubic',
                                     extrapolate='head', border='mean', res=64, size=1, cmap='jet', vlim=(None, None),
                                     cnorm=None, axes=axes[icomp], show=False)
                axes[icomp].set_title(f'Component {icomp + 1}')
                divider = make_axes_locatable(axes[icomp])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = fig.colorbar(axes[icomp].images[-1], cax=cax, shrink=0.6, orientation='vertical')

            plt.savefig(figure_path + f'{subject_id}_{freq_band}_{cond_name}.png')
            plt.close(fig)