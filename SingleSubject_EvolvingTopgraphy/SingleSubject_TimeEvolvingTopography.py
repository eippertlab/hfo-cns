# Testing only in subject 6, median as they have high SNR HFOs

import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
import matplotlib.pyplot as plt
from Common_Functions.IsopotentialFunctions_CbarLabel import mrmr_esg_isopotentialplot
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == '__main__':
    subject = 6
    condition = 2
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    figure_path = '/data/p_02718/Polished/SingleSubject_EvolvingSpatialPatterns/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    # HFO before CCA
    fname = f"sigma_{cond_name}.fif"
    input_path = f"/data/pt_02718/tmp_data/freq_banded_esg/{subject_id}/"
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)

    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    axes = ax.flatten()
    chan_labels = esg_chans
    colorbar_axes = [-0.1, 0.1]
    subjects_4grid = np.arange(1, 37)
    # then the function takes the average over the channel positions of all those subjects
    colorbar = True
    time_points = [8/1000, 8.5/1000, 9/1000, 9.5/1000, 10/1000, 10.5/1000, 11/1000, 11.5/1000]
    for axis, time_point in zip(axes, time_points):
        # Extracts value at time point only
        chanvalues = evoked.copy().pick(esg_chans).get_data(tmin=time_point, tmax=time_point+2/10000)
        mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                  time_point, axis, colorbar_label='Amplitude (\u03BCV)',
                                  srmr_nr=srmr_nr)
        axis.set_yticklabels([])
        axis.set_ylabel(None)
        axis.set_xticklabels([])
        axis.set_xlabel(None)
        axis.set_title(f"{time_point}s")
    plt.tight_layout()

    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    axes = ax.flatten()
    chan_labels = esg_chans
    colorbar_axes = [-0.1, 0.1]
    subjects_4grid = np.arange(1, 37)
    # then the function takes the average over the channel positions of all those subjects
    colorbar = True
    time_points = [12/1000, 12.5/1000, 13/1000, 13.5/1000, 14/1000, 14.5/1000, 15/1000, 15.5/1000]
    for axis, time_point in zip(axes, time_points):
        # Extracts value at time point only
        chanvalues = evoked.copy().pick(esg_chans).get_data(tmin=time_point, tmax=time_point+2/10000)
        mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                  time_point, axis, colorbar_label='Amplitude (\u03BCV)',
                                  srmr_nr=srmr_nr)
        axis.set_yticklabels([])
        axis.set_ylabel(None)
        axis.set_xticklabels([])
        axis.set_xlabel(None)
        axis.set_title(f"{time_point}s")
    plt.tight_layout()
    plt.show()
    # fig.savefig(figure_path + f'{subject_id}_HF_Spatial_{cond_name}')
    # fig.savefig(figure_path + f'{subject_id}_HF_Spatial_{cond_name}.pdf',
    #                 bbox_inches='tight', format="pdf")

