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
    data_types = ['Cortical', 'Thalamic']

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

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated_Digits.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

        elif data_type == 'Thalamic':
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated_Digits.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated_Digits.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)
        else:
            raise RuntimeError('Data type must be Cortical or Thalamic, only')

        figure_path = '/data/p_02718/Polished_2/Digits_Envelope_ClusterTestPerCondition/'
        os.makedirs(figure_path, exist_ok=True)

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
                        epochs = epochs.pick([channel])
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
                        # Data is just a string of values
                        y = cleaned_data[0][x]  # y values to be fitted
                        y_interpol_before = y[x_interpol]
                        y_interpol = pchip(x, y)(x_interpol)  # perform interpolation
                        cleaned_data[0][x_interpol] = y_interpol  # replace in data

                        if trigger_name == 'med1':
                            subj_list.append(subject_id)

                        evoked_list.append(cleaned_data.reshape(-1))

            # Perform test and plot
            times = evoked.times

            for evoked_list, trigger_name in zip([evoked_list_1, evoked_list_2, evoked_list_12], ['finger1', 'finger2', 'finger12']):
                T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                    X=np.array(evoked_list),
                    n_permutations=2000,
                    tail=1,
                    n_jobs=None,
                    out_type="mask",
                    seed=np.random.default_rng(seed=8675309))

                # Plot each individuals difference line instead of ga
                fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
                ax.set_title(f"Envelope")
                for evoked in evoked_list:
                    ax.plot(
                        times,
                        evoked.reshape(-1)
                    )
                ax.set_ylabel("Amplitude")
                ax.set_xlim([0, 0.07])

                for i_c, c in enumerate(clusters):
                    c = c[0]
                    if cluster_p_values[i_c] <= 0.05:
                        h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
                        print(data_type)
                        print('Significant Cluster')
                        print(f'p-val: {cluster_p_values[i_c]}')
                        print(times[c.start])
                        print(times[c.stop-1])
                    else:
                        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)
                        print(data_type)
                        print('Insignificant Cluster')
                        print(f'p-val: {cluster_p_values[i_c]}')
                        print(times[c.start])
                        print(times[c.stop-1])

                hf = plt.plot(times, T_obs, "g")
                # ax2.legend((h,), ("cluster p-value < 0.05",))
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("t-values")
                ax2.set_xlim([0, 0.07])

                plt.savefig(figure_path + f'SingleSubj_Clusters_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}_{trigger_name}.png')
                plt.savefig(figure_path + f'SingleSubj_Clusters_{data_type}_{freq_band}_{cond_name}_n={len(evoked_list)}_{trigger_name}.pdf',
                            bbox_inches='tight', format="pdf")

                # plt.show()
                plt.close()
