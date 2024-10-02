# For digit stimulation (fingers & toes)
# Plot single subject spatial pattern


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions_CbarLabel import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
import pickle
import matplotlib as mpl
import pandas as pd
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Common_Functions.check_excel_exist_component import check_excel_exist
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':

    freq_band = 'sigma'
    srmr_nr = 2

    if srmr_nr != 2:
        print('Error: This script is only designed to work for experiment 2 (digit stimulation)')
        exit()

    subjects = np.arange(1, 25)  # 1 through 24 to access subject data
    conditions = [2, 4]  # Conditions of interest - med_digits and tib_digits
    component_fname = '/data/pt_02718/tmp_data_2/Components_Updated_Digits.xlsx'
    visibility_fname = '/data/pt_02718/tmp_data_2/Visibility_Updated_Digits.xlsx'
    figure_path = '/data/p_02718/Images_2/CCA_digits/ComponentIsopotentialPlots/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for condition in conditions:
        # Set variables
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_names = cond_info.trigger_name  # Will return list of 3, we just want the 12 one
        trigger_name = trigger_names[2]

        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'
            # Spatial pattern
            fname = f"A_st_{freq_band}_{cond_name}.pkl"
            input_path = "/data/pt_02718/tmp_data_2/cca/" + subject_id + "/"
            with open(f'{input_path}{fname}', 'rb') as f:
                A_st = pickle.load(f)

            ####### Isopotential Plots for the first 4 components ########
            # fig, axes = plt.figure()
            fig, axes = plt.subplots(2, 2)
            axes = axes.flatten()
            if cond_name == 'med_digits':
                chan_labels = cervical_chans
                colorbar_axes = [-0.25, 0.25]
            elif cond_name == 'tib_digits':
                chan_labels = lumbar_chans
                colorbar_axes = [-0.2, 0.2]
            for icomp in np.arange(0, 4):  # Plot for each of four components
                subjects_4grid = np.arange(1, 25)  # subj  # Pass this instead of (1, 37) for 1 subjects
                # you can also base the grid on an several subjects
                # then the function takes the average over the channel positions of all those subjects
                time = 0.0
                colorbar = True
                mrmr_esg_isopotentialplot(subjects_4grid, A_st[:, icomp], colorbar_axes, chan_labels,
                                          colorbar, time, axes[icomp], colorbar_label='Amplitude (AU)', srmr_nr=srmr_nr)
                axes[icomp].set_yticklabels([])
                axes[icomp].set_ylabel(None)
                axes[icomp].set_xticklabels([])
                axes[icomp].set_xlabel(None)
                axes[icomp].set_title(f'Component {icomp + 1}')

            plt.savefig(figure_path + f'{subject_id}_{freq_band}_{cond_name}.png')
            plt.close(fig)
