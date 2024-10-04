# Plot grand average envelope after application of CCA on CNS_Level_Specific_Functions data
# Can use only the subjects with visible HFOs, or all subjects regardless


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.invert import invert
from Common_Functions.envelope_noise_reduction import envelope_noise_reduction
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy.stats import sem
from scipy.interpolate import PchipInterpolator as pchip
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 2  # Always 2 for digit information
    data_types = ['Cortical', 'Thalamic']  # 'Thalamic',

    subjects = np.arange(1, 25)
    conditions = [2]  # med_digits
    freq_band = 'sigma'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    for data_type in data_types:
        if data_type == 'Cortical':
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated_Digits.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Images_2/Digits_Envelope_Comparison_SingleSubject/'
            os.makedirs(figure_path, exist_ok=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated_Digits.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

        elif data_type == 'Thalamic':
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated_Digits.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Images_2/Digits_Envelope_Comparison_SingleSubject/'
            os.makedirs(figure_path, exist_ok=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated_Digits.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)
        else:
            raise RuntimeError('Data type must be Cortical or Thalamic, only')

        for condition in conditions:
            # Set variables
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_names = cond_info.trigger_name
            evoked_list_1 = []
            evoked_list_2 = []
            evoked_list_12 = []
            subj_list = []
            for subject in subjects:
                subject_id = f'sub-{str(subject).zfill(3)}'

                ##########################################################
                # Time  Course Information
                ##########################################################
                # Select the right files
                fname = f"{freq_band}_{cond_name}.fif"
                if data_type == 'Cortical':
                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"
                elif data_type == 'Thalamic':
                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg_thalamic/" + subject_id + "/"
                epochs_all = mne.read_epochs(input_path + fname, preload=True)

                for trigger_name, evoked_list in zip(trigger_names, [evoked_list_1, evoked_list_2, evoked_list_12]):
                    epochs = epochs_all[trigger_name]

                    # Need to pick channel based on excel sheet
                    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                    if visible == 'T':
                        channel = f'Cor{int(channel_no)}'
                        inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                        epochs = epochs.pick_channels([channel])
                        if inv == 'T':
                            epochs.apply_function(invert, picks=channel)
                        evoked = epochs.copy().average()
                        evoked.crop(tmin=-0.1, tmax=0.07)
                        envelope = evoked.apply_hilbert(envelope=True)
                        data = envelope.get_data()

                        # Want to subtract mean noise in noise period (-100ms to -10ms) from each data point in the envelope
                        interpol_indices = evoked.time_as_index([-7/1000, 7/1000])
                        noise_data = envelope.copy().crop(-100/1000, -10/1000).get_data()
                        cleaned_data = envelope_noise_reduction(data, noise_data)
                        # Data is interpolated from -7ms to 7ms for stim artefact - do this again to combat subtraction
                        # x is all values EXCEPT those in the interpolation window
                        x_total = np.arange(0, len(evoked.times))
                        x_before = x_total[0:interpol_indices[0]]
                        x_interpol = x_total[interpol_indices[0]:interpol_indices[1]]
                        x_after = x_total[interpol_indices[1]:]
                        x = np.concatenate((x_before, x_after))
                        # # Data is just a string of values
                        y = cleaned_data[0][x]  # y values to be fitted
                        y_interpol_before = y[x_interpol]
                        y_interpol = pchip(x, y)(x_interpol)  # perform interpolation
                        cleaned_data[0][x_interpol] = y_interpol  # replace in data

                        if trigger_name == 'med1':
                            subj_list.append(subject_id)
                        evoked_list.append(cleaned_data)

            # Get list of med1 + med2 curves for each subject
            combined_list = [i[0] + j[0] for i,j in zip(evoked_list_1, evoked_list_2)]
            difference_list = [i - j[0] for i, j in zip(combined_list, evoked_list_12)]

            if data_type == 'Cortical':
                fig_env, axes_env = plt.subplots(5, 4, figsize=(20, 16))
                axes_env = axes_env.flatten()
                fig_diff, axes_diff = plt.subplots(5, 4,  figsize=(20, 16))
                axes_diff = axes_diff.flatten()
            elif data_type == 'Thalamic':
                fig_env, axes_env = plt.subplots(5, 2,  figsize=(20, 16))
                axes_env = axes_env.flatten()
                fig_diff, axes_diff = plt.subplots(5, 2,  figsize=(20, 16))
                axes_diff = axes_diff.flatten()

            # single subject envelopes
            count = 0
            for env_12, env_1plus2, subj in zip(evoked_list_12, combined_list, subj_list):
                axes_env[count].plot(evoked.times, env_12.reshape(-1), label='finger12')
                axes_env[count].plot(evoked.times, env_1plus2, label='finger1 + finger2')
                axes_env[count].set_title(subj)
                axes_env[count].set_xlim([0, 0.07])
                axes_env[count].legend()
                count += 1
            plt.tight_layout()
            fig_env.savefig(figure_path + f'Envelopes_{data_type}_{freq_band}_{cond_name}.png')


            # single subject difference
            count = 0
            for diff, subj in zip(difference_list, subj_list):
                axes_diff[count].plot(evoked.times, diff.reshape(-1))
                axes_diff[count].set_title(subj)
                axes_diff[count].set_xlim([0, 0.07])
                count+=1
            plt.tight_layout()
            fig_diff.savefig(figure_path + f'EnvelopeDifference_{data_type}_{freq_band}_{cond_name}.png')

            plt.show()

