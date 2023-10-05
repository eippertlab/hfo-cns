# Plot potentials for single subjects to find latency of evoked potentials
# CAREFUL: For shifting it should be expected - sep_latency


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.GetTimeToAlign_Old import get_time_to_align
import matplotlib.pyplot as plt
import matplotlib as mpl
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 2
    if srmr_nr == 1:
        subjects = np.arange(1, 37)  # 1 through 36 to access subject data
        conditions = [2, 3]  # Conditions of interest
        figure_path = '/data/p_02718/Images/ESG/LowFrequencyPotentials/'
        os.makedirs(figure_path, exist_ok=True)

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)  # (1, 2) # 1 through 24 to access subject data
        conditions = [3, 5]  # Conditions of interest - med_mixed and tib_mixed
        figure_path = '/data/p_02718/Images_2/ESG/LowFrequencyPotentials/'
        os.makedirs(figure_path, exist_ok=True)

    tstart = 0.00
    tend = 0.07

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    for condition in conditions:
        evoked_list_low = []

        # Set variables
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_name = cond_info.trigger_name

        if cond_name in ['median', 'med_mixed']:
            target_electrode = 'SC6'
        elif cond_name in ['tibial', 'tib_mixed']:
            target_electrode = 'L1'

        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'

            if srmr_nr == 1:
                input_path_low = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                fname_low = f"epochs_{cond_name}.fif"
            elif srmr_nr == 2:
                input_path_low = f"/data/pt_02569/tmp_data_2/ssp_py/{subject_id}/esg/prepro/6 projections/"
                fname_low = f"ssp_cleaned_{cond_name}.fif"

            ##########################################################
            # Time  Course Information
            ##########################################################
            # Create figure
            plt.figure()

            # Low Freq SEP
            if srmr_nr == 1:
                epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
            elif srmr_nr == 2:
                raw = mne.io.read_raw_fif(input_path_low+fname_low, preload=True)
                events, event_ids = mne.events_from_annotations(raw)
                event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                epochs_low = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0],
                                        tmax=iv_epoch[1] - 1 / 1000, baseline=tuple(iv_baseline), preload=True,
                                        reject_by_annotation=True)

            evoked_low = epochs_low.pick_channels([target_electrode]).average()
            evoked_list_low.append(evoked_low.copy().crop(tmin=tstart, tmax=tend))
            plt.plot(evoked_low.copy().crop(tmin=tstart, tmax=tend).times,
                       evoked_low.copy().crop(tmin=tstart, tmax=tend).get_data().reshape(-1))

            plt.suptitle(f'Subject {subject}, {cond_name}')
            plt.legend()
            plt.tight_layout()
            fname = f'{subject_id}_{cond_name}'
            plt.savefig(figure_path+fname)
            plt.show()

        ga = mne.grand_average(evoked_list_low)
        plt.figure()
        plt.plot(ga.times, ga.get_data().reshape(-1))
        plt.legend()
        plt.suptitle(f'Grand Average, {cond_name}')
        fname = f'GrandAverage_{cond_name}'
        plt.tight_layout()
        plt.savefig(figure_path+fname)

        # plt.show()
