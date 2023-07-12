# Perform Oversampled Temporal Projection

import os
import mne
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import *
import matplotlib.pyplot as plt
import pandas as pd
from Common_Functions.evoked_from_raw import evoked_from_raw


def apply_OTP(subject, condition, srmr_nr, sampling_rate, both_patches):
    # set variables
    subject_id = f'sub-{str(subject).zfill(3)}'
    load_path = "/data/pt_02569/tmp_data/prepared_py/" + subject_id + "/esg/prepro/"
    save_path = "/data/pt_02718/tmp_data_otp/otp_cleaned_TestProj1Data/" + subject_id + "/"
    os.makedirs(save_path, exist_ok=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    figure_path = '/data/p_02718/Images_OTP/ESG/TestProject1_LowFrequency/'
    os.makedirs(figure_path, exist_ok=True)

    # get condition info
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    nerve = cond_info.nerve
    nblocks = cond_info.nblocks
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    ###########################################################################################
    # Load
    ###########################################################################################
    # load imported ESG data
    fname = f'noStimart_sr{sampling_rate}_{cond_name}_withqrs_pchip.fif'

    raw = mne.io.read_raw_fif(load_path + fname, preload=True)

    ##########################################################################################
    # OTP
    ##########################################################################################
    if both_patches:
        clean_raw = mne.preprocessing.oversampled_temporal_projection(raw)
    else:
        if cond_name in ['median', 'med_mixed']:
            clean_raw = mne.preprocessing.oversampled_temporal_projection(raw, picks=cervical_chans)
        elif cond_name in ['tibial', 'tib_mixed']:
            clean_raw = mne.preprocessing.oversampled_temporal_projection(raw, picks=lumbar_chans)

    #############################################################################################
    # Make comparison plots for raw versus OTP cleaned data
    #############################################################################################
    median_names = ['median', 'med_mixed']
    tibial_names = ['tibial', 'tib_mixed']
    if cond_name in median_names:
        cluster_channels = ['S6', 'SC6', 'S14']
    elif cond_name in tibial_names:
        cluster_channels = ['S23', 'L1', 'S31']
    fig, ax = plt.subplots()
    count=0
    for data in [raw, clean_raw]:
        if count == 0:
            lab = 'Raw'
        else:
            lab = 'OTP'
        evoked = evoked_from_raw(data, iv_epoch, iv_baseline, trigger_name, False)
        cluster_ix = mne.pick_channels(evoked.info["ch_names"], include=cluster_channels)
        groups = dict(ROI=cluster_ix)
        roi_evoked = mne.channels.combine_channels(evoked, groups, method="mean")
        roi_evoked.crop(tmin=-0.06, tmax=0.07)

        # Plot Time Course
        ax.plot(roi_evoked.times, roi_evoked.get_data().reshape(-1) * 10 ** 6, label=lab)
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Evoked, {subject_id}, channels:{cluster_channels}')
        ax.set_ylabel('Amplitude (uV)')
        count+=1

    if cond_name in median_names:
        ax.set_xlim([0.0, 0.05])
        ax.axvline(x=0.013)
    else:
        ax.set_xlim([0.0, 0.07])
        ax.axvline(x=0.022)

    plt.legend()
    plt.tight_layout()
    if both_patches:
        plt.savefig(figure_path + f'{subject_id}_{cond_name}.png')
    else:
        plt.savefig(figure_path + f'{subject_id}_{cond_name}_separatepatch.png')

    # ##############################################################################################
    # # Save
    # ##############################################################################################
    # # Save the OTP cleaned data for future comparison
    # if both_patches:
    #     clean_raw.save(f"{save_path}otp_cleaned_{cond_name}.fif", fmt='double', overwrite=True)
    # else:
    #     clean_raw.save(f"{save_path}otp_cleaned_{cond_name}_separatepatch.fif", fmt='double', overwrite=True)


if __name__ == '__main__':
    srmr_no = 1
    n_subjects = 36  # Number of subjects
    subjects = [6, 15, 18, 25, 26]  # First 2 I currently reject median, second 2 tibial
    conditions = [3]  # Conditions of interest 2,
    s_rate = 1000  # Refers to sampling rate of the data in project 1
    both_patch = False

    for subj in subjects:
        for cond in conditions:
            apply_OTP(subj, cond, srmr_no, s_rate, both_patch)
