# Plot potentials for single subjects before and after shifting to see what's going wrong here
# Answer seems to be timing estimations from Birgit are not super accurate
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
    use_birgitstiming = False  # If True, use estimates Birgit made

    subjects = np.arange(1, 37)
    conditions = [3, 2]
    freq_bands = ['sigma']
    srmr_nr = 1

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

    figure_path = '/data/p_02718/Images/ESG/LowFrequencyPotentials/'
    os.makedirs(figure_path, exist_ok=True)

    # Spinal Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
    df_vis = pd.read_excel(xls, 'CCA_Spinal')
    df_vis.set_index('Subject', inplace=True)

    xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/Spinal_Timing.xlsx')
    df_timing = pd.read_excel(xls_timing, 'Timing')
    df_timing.set_index('Subject', inplace=True)

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list_low_unshifted = []
            evoked_list_low_shifted = []

            if condition == 2:
                target_electrode = 'SC6'
            elif condition == 3:
                target_electrode = 'L1'

            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                ##########################################################
                # Time  Course Information
                ##########################################################
                # Create figure
                fig, ax = plt.subplots(1, 2)
                # Timing
                if use_birgitstiming:
                    potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
                    fname_pot = 'potential_latency.mat'
                    matdata = loadmat(potential_path + fname_pot)

                # Low Freq SEP
                input_path_low = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                fname_low = f"epochs_{cond_name}.fif"
                epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                evoked_low = epochs_low.pick_channels([target_electrode]).average()
                evoked_list_low_unshifted.append(evoked_low.copy().crop(tmin=tstart, tmax=tend))
                ax[0].plot(evoked_low.copy().crop(tmin=tstart, tmax=tend).times,
                           evoked_low.copy().crop(tmin=tstart, tmax=tend).get_data().reshape(-1))

                # Apply relative time-shift depending on expected latency
                median_lat, tibial_lat = get_time_to_align('esg', ['median', 'tibial'], np.arange(1, 37))
                if cond_name == 'median':
                    if use_birgitstiming:
                        sep_latency = matdata['med_potlatency']
                    else:
                        sep_latency = df_timing.loc[subject, f"N13"]
                    expected = median_lat
                    # expected = 13/1000
                elif cond_name == 'tibial':
                    if use_birgitstiming:
                        sep_latency = matdata['tib_potlatency']
                    else:
                        sep_latency = df_timing.loc[subject, f"N22"]
                    expected = tibial_lat
                    # expected = 22/1000
                if use_birgitstiming:
                    shift = expected - sep_latency[0][0] / 1000
                    ax[0].axvline(sep_latency[0][0] / 1000, color='red', label='Actual Timing')
                    ax[1].axvline(sep_latency[0][0] / 1000, color='red', label='Actual Timing')
                else:
                    shift = expected - sep_latency
                    ax[0].axvline(sep_latency, color='red', label='Actual Timing')
                    ax[1].axvline(sep_latency, color='red', label='Actual Timing')

                evoked_low.shift_time(shift, relative=True)
                evoked_list_low_shifted.append(evoked_low.crop(tmin=tstart, tmax=tend))
                ax[0].axvline(expected, color='green', label='Expected')
                ax[1].axvline(expected, color='green', label='Expected')
                ax[1].plot(evoked_low.crop(tmin=tstart, tmax=tend).times,
                           evoked_low.crop(tmin=tstart, tmax=tend).get_data().reshape(-1))
                fig.suptitle(f'Subject {subject}, {cond_name}')
                ax[0].set_title('Before Shift')
                ax[1].set_title('After Shift')
                plt.legend()
                plt.tight_layout()
                if use_birgitstiming:
                    fname = f'{subject_id}_{cond_name}_BNTiming'
                else:
                    fname = f'{subject_id}_{cond_name}_EBTiming'
                plt.savefig(figure_path+fname)

            unshifted_ga = mne.grand_average(evoked_list_low_unshifted)
            shifted_ga = mne.grand_average(evoked_list_low_shifted)
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(unshifted_ga.times, unshifted_ga.get_data().reshape(-1))
            ax[0].set_title('Unshifted')
            ax[1].plot(shifted_ga.times, shifted_ga.get_data().reshape(-1))
            ax[1].set_title('Shifted')
            ax[0].axvline(expected, color='green', label='Expected Timing')
            ax[1].axvline(expected, color='green', label='Expected Timing')
            plt.legend()
            if use_birgitstiming:
                fig.suptitle(f'Grand Average, {cond_name}_BN')
                fname = f'GrandAverage_{cond_name}_BN'
            else:
                fig.suptitle(f'Grand Average, {cond_name}_EB')
                fname = f'GrandAverage_{cond_name}_EB'
            plt.tight_layout()
            plt.savefig(figure_path+fname)

                # plt.show()
