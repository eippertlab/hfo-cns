# Plot grand average envelope after application of CCA on EEG data
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
    exclude_outlier = False  # If true, exclude subj 1 in case of cortical
    shorter_time = True  # If true, crop the epoch before testing

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

            figure_path = '/data/p_02718/Polished_2/Digits_Envelope_Comparison/'
            os.makedirs(figure_path, exist_ok=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated_Digits.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

        elif data_type == 'Thalamic':
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated_Digits.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Polished_2/Digits_Envelope_Comparison/'
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
                        if shorter_time:
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

                        if data_type == 'Cortical' and exclude_outlier:
                            if subject_id != 'sub-001':
                                if trigger_name == 'med1':
                                    subj_list.append(subject_id)
                                evoked_list.append(cleaned_data)
                        else:
                            if trigger_name == 'med1':
                                subj_list.append(subject_id)
                            evoked_list.append(cleaned_data)

            # Get list of med1 + med2 curves for each subject
            combined_list = [i[0] + j[0] for i,j in zip(evoked_list_1, evoked_list_2)]

            # Get grand average across chosen epochs, and spatial patterns
            grand_average_1 = np.mean(evoked_list_1, axis=0)
            grand_average_2 = np.mean(evoked_list_2, axis=0)
            grand_average_12 = np.mean(evoked_list_12, axis=0)
            grand_average_1plus2 = np.mean(combined_list, axis=0).reshape(-1, 1).T

            # Get error bands across subjects
            upper_list = []
            lower_list =[]
            for evoked_list, grand_average in zip([evoked_list_1, evoked_list_2, evoked_list_12, combined_list],
                                                  [grand_average_1, grand_average_2, grand_average_12, grand_average_1plus2]):
                error = sem(evoked_list, axis=0)
                upper_list.append((grand_average[0, :] + error).reshape(-1))
                lower_list.append((grand_average[0, :] - error).reshape(-1))

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(evoked.times, grand_average_1[0, :], label='med1')
            ax.plot(evoked.times, grand_average_2[0, :], label='med2')
            ax.plot(evoked.times, grand_average_12[0, :], label='med12')
            ax.plot(evoked.times, grand_average_1plus2[0, :], label='med1+med2')
            for lower, upper in zip(lower_list, upper_list):
                ax.fill_between(evoked.times, lower, upper, alpha=0.3)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
            if cond_name == 'median':
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])

            plt.legend()
            plt.tight_layout()
            plt.savefig(figure_path+f'GA_Envelope_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}.png')
            plt.savefig(figure_path+f'GA_Envelope_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                        bbox_inches='tight', format="pdf")

            # Run permutation cluster test
            # combined_array = np.array(combined_list)
            # [combined_array[:, np.newaxis, :], np.array(evoked_list_12)]
            # T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_test(
            #     [np.array(np.squeeze(evoked_list_12)), np.array(combined_list)],
            #     n_permutations=2000,
            #     tail=1,
            #     n_jobs=None,
            #     out_type="mask",)
            # Run permutation cluster test
            # combined_array = np.array(combined_list)
            # [combined_array[:, np.newaxis, :], np.array(evoked_list_12)]
            difference_list = [i[0] - j[0] for i, j in zip(evoked_list_12, combined_list)]
            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                X=np.array(difference_list),
                n_permutations=2000,
                tail=1,
                n_jobs=None,
                out_type="mask", )

            grand_average_difference = np.mean(difference_list, axis=0).reshape(-1, 1).T
            error = sem(difference_list, axis=0)
            upper = (grand_average_difference[0, :] + error).reshape(-1)
            lower = (grand_average_difference[0, :] - error).reshape(-1)
            times = evoked.times
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
            ax.set_title(f"{data_type}, Difference")
            ax.plot(
                times,
                grand_average_difference[0, :],
                label="Envelopes med12 - (med1+med2)",
            )
            ax.fill_between(evoked.times, lower, upper, alpha=0.3)
            ax.set_ylabel("Amplitude")
            ax.set_xlim([0, 0.07])
            ax.legend()

            for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                else:
                    ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

            hf = plt.plot(times, T_obs, "g")
            # ax2.legend((h,), ("cluster p-value < 0.05",))
            ax2.set_xlabel("time (ms)")
            ax2.set_ylabel("f-values")
            ax2.set_xlim([0, 0.07])

            plt.savefig(figure_path + f'GA_Clusters_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}.png')
            plt.savefig(figure_path + f'GA_Clusters_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                        bbox_inches='tight', format="pdf")

            # Plot each individuals difference line instead of ga
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
            ax.set_title(f"{data_type}, Difference")
            for evoked, subj in zip(difference_list, subj_list):
                ax.plot(
                    times,
                    evoked,
                    label=f"{subj}",
                )
            ax.set_ylabel("Amplitude")
            ax.set_xlim([0, 0.07])

            for i_c, c in enumerate(clusters):
                c = c[0]
                if cluster_p_values[i_c] <= 0.05:
                    h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                    print(data_type)
                    print('Significant Cluster')
                    print(times[c.start])
                    print(times[c.stop])
                else:
                    ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)
                    print(data_type)
                    print('Insignificant Cluster')
                    print(times[c.start])
                    print(times[c.stop])

            hf = plt.plot(times, T_obs, "g")
            # ax2.legend((h,), ("cluster p-value < 0.05",))
            ax2.set_xlabel("time (ms)")
            ax2.set_ylabel("f-values")
            ax2.set_xlim([0, 0.07])

            plt.savefig(figure_path + f'SingleSubj_Clusters_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}.png')
            plt.savefig(figure_path + f'SingleSubj_Clusters_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                        bbox_inches='tight', format="pdf")

            plt.show()
