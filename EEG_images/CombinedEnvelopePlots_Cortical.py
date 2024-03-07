# Want to plot envelope from a single electrode, cluster electrode and envelope from the CCA data on a YY plot
# Want to add error bands to show standard error of the mean in same

import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.evoked_from_raw import evoked_from_raw
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy.stats import sem
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    for srmr_nr in [1,2]:
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
            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

            xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
            df_timing = pd.read_excel(xls_timing, 'Cortical')
            df_timing.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Images/EEG&CCA/Combined_Envelopes/'
            os.makedirs(figure_path, exist_ok=True)
        elif srmr_nr == 2:
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

            xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx')
            df_timing = pd.read_excel(xls_timing, 'Cortical')
            df_timing.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Images_2/EEG&CCA/Combined_Envelopes/'
            os.makedirs(figure_path, exist_ok=True)

        median_names = ['median', 'med_mixed']
        tibial_names = ['tibial', 'tib_mixed']

        for freq_band in freq_bands:
            for condition in conditions:
                fig, ax1 = plt.subplots()
                ax10 = ax1.twinx()
                for data_type in ['single_electrode', 'CCA']:
                    evoked_list = []
                    for subject in subjects:
                        # Set variables
                        cond_info = get_conditioninfo(condition, srmr_nr)
                        cond_name = cond_info.cond_name
                        trigger_name = cond_info.trigger_name
                        subject_id = f'sub-{str(subject).zfill(3)}'

                        ################################################################################################
                        # Only perform any of the following if this subject is marked as included
                        ################################################################################################
                        visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                        if visible == 'T':
                            ############################################################################################
                            # Get time to shift for this subject and set the electrode options for raw ESG
                            ############################################################################################
                            if srmr_nr == 1:
                                condition_names = ['median', 'tibial']
                            elif srmr_nr == 2:
                                condition_names = ['med_mixed', 'tib_mixed']
                            if cond_name in median_names:
                                sep_latency = round(df_timing.loc[subject, f"N20"], 3)
                                expected = 0.02
                                if data_type == 'cluster_electrodes':
                                    channels = ['CP4', 'C4', 'CP6', 'P4', 'CP2']
                                elif data_type == 'single_electrode':
                                    channels = ['CP4']
                            elif cond_name in tibial_names:
                                sep_latency = round(df_timing.loc[subject, f"P39"], 3)
                                expected = 0.04
                                if data_type == 'cluster_electrodes':
                                    channels = ['Cz', 'FCz', 'C2', 'CPz', 'C1']
                                elif data_type == 'single_electrode':
                                    channels = ['Cz']
                            shift = expected - sep_latency

                            ######################################################################################
                            # Select the right filenames and filepaths, read in and get in evoked form
                            ######################################################################################
                            fname = f"{freq_band}_{cond_name}.fif"
                            if data_type in ['single_electrode', 'cluster_electrodes']:
                                if srmr_nr == 1:
                                    input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
                                elif srmr_nr == 2:
                                    input_path = "/data/pt_02718/tmp_data_2/freq_banded_eeg/" + subject_id + "/"
                                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                                evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                            else:
                                if srmr_nr == 1:
                                    input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                                elif srmr_nr == 2:
                                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"
                                epochs = mne.read_epochs(input_path + fname, preload=True)
                                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                                channels = f'Cor{channel_no}'
                                inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                                epochs = epochs.pick_channels([channels])
                                if inv == 'T':
                                    epochs.apply_function(invert, picks=channels)
                                evoked = epochs.copy().average()

                            ############################################################################################
                            # Compute the actual envelopes
                            ############################################################################################
                            if data_type == 'single_electrode':
                                evoked = evoked.pick(channels)
                                evoked.shift_time(shift, relative=True)
                                evoked.crop(tmin=-0.06, tmax=0.07)
                                envelope = evoked.apply_hilbert(envelope=True)
                            elif data_type == 'cluster_electrodes':
                                cluster_ix = mne.pick_channels(evoked.info["ch_names"], include=channels)
                                groups = dict(ROI=cluster_ix)
                                roi_evoked = mne.channels.combine_channels(evoked, groups, method="mean")
                                roi_evoked.shift_time(shift, relative=True)
                                roi_evoked.crop(tmin=-0.06, tmax=0.07)
                                envelope = roi_evoked.apply_hilbert(envelope=True)
                            else:  # For CCA, at this point the channel selection is already done prior to making evoked
                                evoked.shift_time(shift, relative=True)
                                evoked.crop(tmin=-0.06, tmax=0.07)
                                envelope = evoked.apply_hilbert(envelope=True)
                            data = envelope.get_data()
                            evoked_list.append(data)

                    #################################################################################################
                    # Get grand average across chosen epochs and the standard error of the mean
                    #################################################################################################
                    grand_average = np.mean(evoked_list, axis=0)
                    error = sem(evoked_list, axis=0)
                    upper = (grand_average[0, :] + error).reshape(-1)
                    lower = (grand_average[0, :] - error).reshape(-1)

                    #################################################################################################
                    # Plot Time Course
                    #################################################################################################
                    if data_type == 'cluster_electrodes':
                        ax1.plot(roi_evoked.times, grand_average[0, :], color='orange', label='Cluster')
                        ax1.fill_between(roi_evoked.times, lower, upper, alpha=0.3, color='orange')
                    elif data_type == 'single_electrode':
                        ax1.plot(evoked.times, grand_average[0, :], color='blue', label='Single')
                        ax1.fill_between(evoked.times, lower, upper, alpha=0.3, color='blue')
                    else:
                        ax10.plot(evoked.times, grand_average[0, :], color='green', label='CCA')
                        ax10.fill_between(evoked.times, lower, upper, alpha=0.3, color='green')

                ax1.set_xlabel('Time (s)')
                ax1.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
                ax1.set_ylabel('Amplitude Raw ESG')
                ax10.set_ylabel('Amplitude After CCA')
                if cond_name in median_names:
                    ax1.set_xlim([0.0, 0.05])
                    ax1.axvline(expected, color='red')
                else:
                    ax1.set_xlim([0.0, 0.07])
                    ax1.axvline(expected, color='red')

                handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(),
                                                           ax10.get_legend_handles_labels())]
                fig.legend(handles, labels)
                plt.tight_layout()
                plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}')
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                            bbox_inches='tight', format="pdf")
                plt.close(fig)
