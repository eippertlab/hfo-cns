# Want to read in from frequency banded signals with or without OTP processing
# Plot single subject evoked specific clusters of electrodes

import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.evoked_from_raw import evoked_from_raw
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 1
    subjects = [6, 15, 18, 25, 26]
    conditions = [2, 3]
    freq_bands = ['sigma']
    both_patches = False

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    figure_path = '/data/p_02718/Images_OTP/ESG/Evoked_SingleSubject_OTPvsRaw/'
    os.makedirs(figure_path, exist_ok=True)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    median_names = ['median', 'med_mixed']
    tibial_names = ['tibial', 'tib_mixed']

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list = []
            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                ##########################################################
                # Time  Course Information
                ##########################################################
                # Align subject with average latency across all subjects
                if srmr_nr == 1:
                    condition_names = ['median', 'tibial']
                elif srmr_nr == 2:
                    condition_names = ['med_mixed', 'tib_mixed']
                if cond_name in median_names:
                    cluster_channels = ['S6', 'SC6', 'S14']
                elif cond_name in tibial_names:
                    cluster_channels = ['S23', 'L1', 'S31']

                # Select the right files
                fname = f"{freq_band}_{cond_name}.fif"
                if both_patches:
                    fname_otp = f"{freq_band}_{cond_name}.fif"
                else:
                    fname_otp = f"{freq_band}_{cond_name}_separatepatch.fif"

                input_path_raw = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
                input_path_otp = "/data/pt_02718/tmp_data_otp/freq_banded_esg/" + subject_id + "/"

                fig, ax = plt.subplots()
                for input_path in [input_path_raw, input_path_otp]:
                    if input_path == input_path_raw:
                        lab = 'Original'
                        raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                    else:
                        lab = 'OTP'
                        raw = mne.io.read_raw_fif(input_path + fname_otp, preload=True)
                    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                    cluster_ix = mne.pick_channels(evoked.info["ch_names"], include=cluster_channels)
                    groups = dict(ROI=cluster_ix)
                    roi_evoked = mne.channels.combine_channels(evoked, groups, method="mean")
                    roi_evoked.crop(tmin=-0.06, tmax=0.07)

                    # Plot Time Course
                    ax.plot(roi_evoked.times, roi_evoked.get_data().reshape(-1)*10**6, label=lab)
                    ax.set_xlabel('Time (s)')
                    ax.set_title(f'Evoked, {subject_id}, channels:{cluster_channels}')
                    ax.set_ylabel('Amplitude (uV)')

                if cond_name in median_names:
                    ax.set_xlim([0.0, 0.05])
                else:
                    ax.set_xlim([0.0, 0.07])

                plt.legend()
                plt.tight_layout()
                if both_patches:
                    plt.savefig(figure_path+f'{subject_id}_{cond_name}.png')
                else:
                    plt.savefig(figure_path+f'{subject_id}_{cond_name}_separatepatch.png')
                # plt.savefig(figure_path + f'{subject_id}_{cond_name}.pdf',
                #             bbox_inches='tight', format="pdf")
                plt.close(fig)

