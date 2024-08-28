# Want to plot envelope from CCA when trained for thalamic versus cortical time windows
# Want to add error bands to show standard error of the mean in same
# Shift only the spinal data

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
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

mpl.rc('font', size=SMALL_SIZE)          # controls default text sizes
mpl.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
mpl.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
mpl.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
mpl.rc('legend', fontsize=10)    # legend fontsize
mpl.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title



if __name__ == '__main__':
    for srmr_nr in [1, 2]:

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

        median_names = ['median', 'med_mixed']
        tibial_names = ['tibial', 'tib_mixed']

        for freq_band in freq_bands:
            for condition in conditions:
                fig, ax1 = plt.subplots()
                for data_type in ['Spinal_CCA', 'Thalamic_CCA', 'Cortical_CCA']:
                    if srmr_nr == 1:
                        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
                        df_timing = pd.read_excel(xls_timing, 'Spinal')
                        df_timing.set_index('Subject', inplace=True)
                        if data_type == 'Spinal_CCA':
                            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Updated.xlsx')
                            df = pd.read_excel(xls, 'CCA')
                            df.set_index('Subject', inplace=True)

                            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
                            df_vis = pd.read_excel(xls, 'CCA_Spinal')
                            df_vis.set_index('Subject', inplace=True)

                        elif data_type == 'Thalamic_CCA':
                            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Thalamic_Updated.xlsx')
                            df = pd.read_excel(xls, 'CCA')
                            df.set_index('Subject', inplace=True)

                            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Thalamic_Updated.xlsx')
                            df_vis = pd.read_excel(xls, 'CCA_Brain')
                            df_vis.set_index('Subject', inplace=True)
                        else:
                            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
                            df = pd.read_excel(xls, 'CCA')
                            df.set_index('Subject', inplace=True)

                            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
                            df_vis = pd.read_excel(xls, 'CCA_Brain')
                            df_vis.set_index('Subject', inplace=True)

                        figure_path = '/data/p_02718/Polished/Spinal_Subcortical_Cortical/'
                        os.makedirs(figure_path, exist_ok=True)

                    elif srmr_nr == 2:
                        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx')
                        df_timing = pd.read_excel(xls_timing, 'Spinal')
                        df_timing.set_index('Subject', inplace=True)
                        if data_type == 'Spinal_CCA':
                            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_Updated.xlsx')
                            df = pd.read_excel(xls, 'CCA')
                            df.set_index('Subject', inplace=True)

                            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated.xlsx')
                            df_vis = pd.read_excel(xls, 'CCA_Spinal')
                            df_vis.set_index('Subject', inplace=True)

                        elif data_type == 'Thalamic_CCA':
                            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated.xlsx')
                            df = pd.read_excel(xls, 'CCA')
                            df.set_index('Subject', inplace=True)

                            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated.xlsx')
                            df_vis = pd.read_excel(xls, 'CCA_Brain')
                            df_vis.set_index('Subject', inplace=True)
                        else:
                            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated.xlsx')
                            df = pd.read_excel(xls, 'CCA')
                            df.set_index('Subject', inplace=True)

                            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated.xlsx')
                            df_vis = pd.read_excel(xls, 'CCA_Brain')
                            df_vis.set_index('Subject', inplace=True)

                        figure_path = '/data/p_02718/Polished_2/Spinal_Subcortical_Cortical/'
                        os.makedirs(figure_path, exist_ok=True)

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
                                if data_type == 'Spinal_CCA':
                                    sep_latency = round(df_timing.loc[subject, f"N13"], 3)
                                    expected = 0.013
                                    shift = expected - sep_latency
                                elif data_type == 'Thalamic_CCA':
                                    expected = 0.014
                                elif data_type == 'Cortical_CCA':
                                    expected = 0.020
                            elif cond_name in tibial_names:
                                if data_type == 'Spinal_CCA':
                                    sep_latency = round(df_timing.loc[subject, f"N22"], 3)
                                    expected = 0.022
                                    shift = expected - sep_latency
                                elif data_type == 'Thalamic_CCA':
                                    expected = 0.030
                                elif data_type == 'Cortical_CCA':
                                    expected = 0.040

                            ######################################################################################
                            # Select the right filenames and filepaths, read in and get in evoked form
                            ######################################################################################
                            fname = f"{freq_band}_{cond_name}.fif"
                            if srmr_nr == 1:
                                if data_type == 'Spinal_CCA':
                                    input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                                elif data_type == 'Thalamic_CCA':
                                    input_path = "/data/pt_02718/tmp_data/cca_eeg_thalamic/" + subject_id + "/"
                                else:
                                    input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                            elif srmr_nr == 2:
                                if data_type == 'Spinal_CCA':
                                    input_path = "/data/pt_02718/tmp_data_2/cca/" + subject_id + "/"
                                elif data_type == 'Thalamic_CCA':
                                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg_thalamic/" + subject_id + "/"
                                else:
                                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"
                            epochs = mne.read_epochs(input_path + fname, preload=True)
                            channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                            channels = f'Cor{channel_no}'
                            inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                            epochs = epochs.pick_channels([channels])
                            if inv == 'T':
                                epochs.apply_function(invert, picks=channels)
                            evoked = epochs.copy().average()
                            if data_type == 'Spinal_CCA':
                                evoked.shift_time(shift, relative=True)

                            ############################################################################################
                            # Compute the actual envelopes
                            ############################################################################################
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

                    # Get latency of envelope across subjects
                    index_max = np.argmax(grand_average)
                    print(f'Study{srmr_nr}, {data_type}, peak latency: {evoked.times[index_max]:0.4f}s')

                    #################################################################################################
                    # Plot Time Course
                    #################################################################################################
                    if data_type == 'Spinal_CCA':
                        color = 'tab:blue'
                        label = f'Spinal, n={len(evoked_list)}'
                    elif data_type == 'Thalamic_CCA':
                        color = 'tab:cyan'
                        label = f'Subcortical, n={len(evoked_list)}'
                    elif data_type == 'Cortical_CCA':
                        color = 'tab:purple'
                        label = f'Cortical, n={len(evoked_list)}'
                    ax1.plot(evoked.times, grand_average[0, :], label=label, color=color)
                    ax1.fill_between(evoked.times, lower, upper, color=color, alpha=0.3)
                    # ax1.axvline(sep_latency, color='red')

                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Amplitude (AU)')
                if cond_name in median_names:
                    ax1.set_xlim([0.0, 0.05])
                    # ax1.axvline(sep_latency, color='red')

                else:
                    ax1.set_xlim([0.0, 0.07])
                    # ax1.axvline(sep_latency, color='red')

                plt.legend(loc='upper right')
                plt.tight_layout()
                # plt.show()
                plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}')
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}.pdf',
                            bbox_inches='tight', format="pdf")
                plt.close(fig)
