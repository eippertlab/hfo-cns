# Want to read in from frequency banded signals
# Plot envelope of grand average of the ESG data in specific clusters of electrodes
# Shifting the envelope based on my updated estimates of the spinal timing

import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.evoked_from_raw import evoked_from_raw
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 2
    cluster_electrodes = False

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        freq_bands = ['sigma']
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        freq_bands = ['sigma']

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    if srmr_nr == 1:
        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
        df_timing = pd.read_excel(xls_timing, 'Cortical')
        df_timing.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/EEG/ClusterElectrode_Envelopes/'
        os.makedirs(figure_path, exist_ok=True)
    elif srmr_nr == 2:
        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx')
        df_timing = pd.read_excel(xls_timing, 'Cortical')
        df_timing.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images_2/EEG/ClusterElectrode_Envelopes/'
        os.makedirs(figure_path, exist_ok=True)

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
                eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

                ##########################################################
                # Time  Course Information
                ##########################################################
                # Align subject with average latency across all subjects
                if srmr_nr == 1:
                    condition_names = ['median', 'tibial']
                elif srmr_nr == 2:
                    condition_names = ['med_mixed', 'tib_mixed']
                if cond_name in median_names:
                    sep_latency = round(df_timing.loc[subject, f"N20"], 3)
                    expected = 0.02
                    if cluster_electrodes:
                        cluster_channels = ['CP4', 'C4', 'CP6', 'P4', 'CP2']
                    else:
                        cluster_channels = ['CP4']
                elif cond_name in tibial_names:
                    sep_latency = round(df_timing.loc[subject, f"P39"], 3)
                    expected = 0.04
                    if cluster_electrodes:
                        cluster_channels = ['Cz', 'FCz', 'C2', 'CPz', 'C1']
                    else:
                        cluster_channels = ['Cz']
                shift = expected - sep_latency

                # Select the right files
                fname = f"{freq_band}_{cond_name}.fif"
                if srmr_nr == 1:
                    input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
                elif srmr_nr == 2:
                    input_path = "/data/pt_02718/tmp_data_2/freq_banded_eeg/" + subject_id + "/"

                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                if cluster_electrodes:
                    cluster_ix = mne.pick_channels(evoked.info["ch_names"], include=cluster_channels)
                    groups = dict(ROI=cluster_ix)
                    roi_evoked = mne.channels.combine_channels(evoked, groups, method="mean")
                    roi_evoked.shift_time(shift, relative=True)
                    roi_evoked.crop(tmin=-0.06, tmax=0.07)
                    envelope = roi_evoked.apply_hilbert(envelope=True)
                else:
                    evoked = evoked.pick(cluster_channels)
                    evoked.shift_time(shift, relative=True)
                    evoked.crop(tmin=-0.06, tmax=0.07)
                    envelope = evoked.apply_hilbert(envelope=True)
                data = envelope.get_data()
                evoked_list.append(data)

            # Get grand average across chosen epochs
            grand_average = np.mean(evoked_list, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            if cluster_electrodes:
                ax.plot(roi_evoked.times, grand_average[0, :])
            else:
                ax.plot(evoked.times, grand_average[0, :])
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}, channels:{cluster_channels}')
            ax.set_ylabel('Amplitude')
            if cond_name in median_names:
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])

            if cluster_electrodes:
                plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}')
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                            bbox_inches='tight', format="pdf")
            else:
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}_single')
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}_single.pdf',
                            bbox_inches='tight', format="pdf")
            plt.close(fig)

