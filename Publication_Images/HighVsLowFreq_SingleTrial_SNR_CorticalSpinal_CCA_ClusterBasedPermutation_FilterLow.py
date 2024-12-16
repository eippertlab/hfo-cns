# Load in the previously computed low and high freuqency snr
# First check if the snr in top vs bottom k% of trials is different
# snr_high and snr_low can still have nan values if there was no neg value for low freq data when searching peak
# First put into dataframe and exclude nan trials

import pickle
import os
import mne
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import sem
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.invert import invert
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
pd.set_option('display.max_rows', 2000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


if __name__ == '__main__':
    data_types = ['Spinal', 'Cortical']  # Can be Cortical or Spinal, not both
    long = True  # Long: -100ms to 300ms, otherwise 0ms to 70ms
    # data_types = ['Cortical']

    srmr_nr = 1
    sfreq = 5000
    n_trials = 200 # Number of trials at top/bottom to test
    freq_band = 'sigma'

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
        fig_folder = 'Polished'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'
        fig_folder = 'Polished_2'

    # Cortical Excel file
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    # Cortical Excel file - low frequency
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated_LF.xlsx')
    df_cortical_lf = pd.read_excel(xls, 'CCA')
    df_cortical_lf.set_index('Subject', inplace=True)

    # Spinal Excel file
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA')
    df_spinal.set_index('Subject', inplace=True)

    # Spinal Excel file - low frequency
    xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated_LF.xlsx')
    df_spinal_lf = pd.read_excel(xls, 'CCA')
    df_spinal_lf.set_index('Subject', inplace=True)

    xls_timing = pd.ExcelFile(f'/data/pt_02718/{folder}/LowFreq_HighFreq_Relation.xlsx')
    df_timing = pd.read_excel(xls_timing, 'Spinal')
    df_timing.set_index('Subject', inplace=True)

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    # End of interpolation for cortical and spinal data added to period before stim starts in epoch
    # This is so that cluster based permutation test does not include interpolation interval
    tmax_esg = 0.007 + abs(iv_epoch[0])
    tmax_eeg = 0.006 + abs(iv_epoch[0])

    for data_type in data_types:
        if data_type == 'Cortical':
            tmax = tmax_eeg
        elif data_type == 'Spinal':
            tmax = tmax_esg

        df_topbottom10 = pd.DataFrame()
        figure_path_highlow = f'/data/p_02718/{fig_folder}/SingleTrialSNR_LowVsHigh_CCA/StrongestVsWeakest_ClusterBasedPermutation_LowFilter/{data_type}/'
        os.makedirs(figure_path_highlow, exist_ok=True)

        for condition in conditions:  # Conditions (median, tibial)
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            top_low = []
            top_high = []
            bottom_low = []
            bottom_high = []
            evoked_list_low_bottom = []
            evoked_list_low_top = []
            evoked_list_high_bottom = []
            evoked_list_high_top = []
            evoked_list_difference_high = []
            evoked_list_difference_low = []

            for subject in subjects:  # All subjects
                subject_id = f'sub-{str(subject).zfill(3)}'

                # For each subject, take the indices of the trials in their sorted form and plot the LF-SEP and
                # HFOs when using the top and bottom 200 trials
                if data_type == 'Cortical':
                    if cond_name in ['median', 'med_mixed']:
                        expected = 0.02
                    else:
                        expected = 0.04
                    # HFO
                    input_path = f"/data/pt_02718/{folder}/cca_eeg/{subject_id}/"
                    df = df_cortical

                    # Low Freq SEP
                    input_path_low = f"/data/pt_02718/{folder}/cca_eeg_low/{subject_id}/"
                    fname_low = f"noStimart_sr5000_{cond_name}_withqrs_eeg.fif"
                    df_low = df_cortical_lf

                elif data_type == 'Spinal':
                    # HFO
                    input_path = f"/data/pt_02718/{folder}/cca/{subject_id}/"
                    df = df_spinal

                    # Low Freq SEP
                    input_path_low = f"/data/pt_02718/{folder}/cca_low/{subject_id}/"
                    fname_low = f"ssp6_cleaned_{cond_name}.fif"
                    df_low = df_spinal_lf

                # Get correct channel for cca HFO data
                fname = f"{freq_band}_{cond_name}.fif"
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if channel_no != 0:
                    channel_cca = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = mne.read_epochs(input_path + fname, preload=True)
                    epochs = epochs.pick([channel_cca])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel_cca)

                # Get correct channel for cca LF data
                channel_no_low = df_low.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if channel_no_low != 0:
                    channel_cca = f'Cor{channel_no_low}'
                    inv = df_low.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                    epochs_low = epochs_low.pick([channel_cca])
                    if inv == 'T':
                        epochs_low.apply_function(invert, picks=channel_cca)

                # Only do analysis if visible component for LF and HFO data
                if channel_no != 0 and channel_no_low != 0:
                    # Filter low freq signals
                    epochs_low.filter(l_freq=10, h_freq=None, method='iir',
                                      iir_params={'order': 5, 'ftype': 'butter'}, phase='zero')
                    df_sub = pd.DataFrame()
                    input_path_snr = f'/data/pt_02718/{folder}/singletrial_snr_cca_filterlow/{subject_id}/'
                    fname_low = f'snr_low_{freq_band}_{cond_name}_{data_type.lower()}.pkl'
                    fname_high = f'snr_high_{freq_band}_{cond_name}_{data_type.lower()}.pkl'

                    with open(f'{input_path_snr}{fname_low}', 'rb') as f:
                        snr_low = pickle.load(f)

                    with open(f'{input_path_snr}{fname_high}', 'rb') as f:
                        snr_high = pickle.load(f)

                    df_sub[f'low'] = snr_low
                    df_sub[f'high'] = snr_high
                    df_sub.dropna(inplace=True)

                    # Sort based on SNR of high frequency trials, then get average SNR across top and bottom 10% of trials
                    df_sub.sort_values('high', inplace=True)
                    # print(df_sub)
                    bottom_low.append(df_sub[:n_trials].mean()['low'])
                    bottom_high.append(df_sub[:n_trials].mean()['high'])
                    top_low.append(df_sub[-n_trials:].mean()['low'])
                    top_high.append(df_sub[-n_trials:].mean()['high'])

                    index_low = df_sub.index[:200].tolist()
                    index_high = df_sub.index[-200:].tolist()
                    # Also need to be wary the srate of LF files is 1kHz, but HF files are 5kHz
                    hf_times = np.linspace(iv_epoch[0], iv_epoch[1]-1/1000, 1996)

                    # Add evoked LF-SEP and envelope of HFO to evoked lists to later create GA image
                    evoked_list_low_bottom.append(epochs_low[index_low].average().data.reshape(-1))
                    evoked_list_low_top.append(epochs_low[index_high].average().data.reshape(-1))
                    evoked_list_high_bottom.append(epochs[index_low].average().apply_hilbert(envelope=True).data.reshape(-1))
                    evoked_list_high_top.append(epochs[index_high].average().apply_hilbert(envelope=True).data.reshape(-1))

                else:
                    bottom_low.append(np.nan)
                    bottom_high.append(np.nan)
                    top_low.append(np.nan)
                    top_high.append(np.nan)

            df_topbottom10[f'{data_type}_{cond_name}_bottom10_low'] = bottom_low
            df_topbottom10[f'{data_type}_{cond_name}_bottom10_high'] = bottom_high
            df_topbottom10[f'{data_type}_{cond_name}_top10_low'] = top_low
            df_topbottom10[f'{data_type}_{cond_name}_top10_high'] = top_high

            # Generate GA info
            ga_low_bottom = np.mean(evoked_list_low_bottom, axis=0)
            ga_low_top = np.mean(evoked_list_low_top, axis=0)
            error_low_bottom = sem(evoked_list_low_bottom, axis=0)
            upper_low_bottom = (ga_low_bottom[:] + error_low_bottom).reshape(-1)
            lower_low_bottom = (ga_low_bottom[:] - error_low_bottom).reshape(-1)
            error_low_top = sem(evoked_list_low_top, axis=0)
            upper_low_top = (ga_low_top[:] + error_low_top).reshape(-1)
            lower_low_top = (ga_low_top[:] - error_low_top).reshape(-1)

            ga_high_bottom = np.mean(evoked_list_high_bottom, axis=0)
            ga_high_top = np.mean(evoked_list_high_top, axis=0)
            error_high_bottom = sem(evoked_list_high_bottom, axis=0)
            upper_high_bottom = (ga_high_bottom[:] + error_high_bottom).reshape(-1)
            lower_high_bottom = (ga_high_bottom[:] - error_high_bottom).reshape(-1)
            error_high_top = sem(evoked_list_high_top, axis=0)
            upper_high_top = (ga_high_top[:] + error_high_top).reshape(-1)
            lower_high_top = (ga_high_top[:] - error_high_top).reshape(-1)

            ################################################################################################
            # Perform cluster based permutation test and plot image
            ################################################################################################
            class ClusterResults():
                def __init__(self):
                    pass
            # Need one sample test as our data is paired - therefore need to get difference in each case
            # Low frequency
            for evoked_top, evoked_bottom in zip(evoked_list_low_top, evoked_list_low_bottom):
                evoked_difference = evoked_top - evoked_bottom
                evoked_list_difference_low.append(evoked_difference[int(sfreq*tmax):])

            # High frequency
            for evoked_top, evoked_bottom in zip(evoked_list_high_top, evoked_list_high_bottom):
                evoked_difference = evoked_top - evoked_bottom
                evoked_list_difference_high.append(evoked_difference[int(sfreq*tmax):])

            cluster_res = ClusterResults()
            for evoked_list_difference, data_name in zip([evoked_list_difference_low, evoked_list_difference_high],
                                                         ['low', 'high']):
                if data_name == 'low':
                    if data_type == 'Cortical' and cond_name in ['tibial', 'tib_mixed']:
                        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                            np.array(evoked_list_difference),
                            out_type="mask",
                            n_permutations=1000,
                            tail=1,
                            seed=np.random.default_rng(seed=8675309),
                        )  # tail = 1 for one-sided test (P40 greater for strongest trials)
                    else:
                        T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                            np.array(evoked_list_difference),
                            out_type="mask",
                            n_permutations=1000,
                            tail=-1,
                            seed=np.random.default_rng(seed=8675309),
                        )  # tail = -1 for one-sided test (N13, N22, N20 smaller for strongest trials)

                    cluster_res.T_obs_low = T_obs
                    cluster_res.clusters_low = clusters
                    cluster_res.cluster_p_values_low = cluster_p_values
                    cluster_res.H0_low = H0
                elif data_name == 'high':
                    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                        np.array(evoked_list_difference),
                        out_type="mask",
                        n_permutations=1000,
                        tail=1,
                        seed=np.random.default_rng(seed=8675309),
                    )  # tail = 1 for one-sided test
                    cluster_res.T_obs_high = T_obs
                    cluster_res.clusters_high = clusters
                    cluster_res.cluster_p_values_high = cluster_p_values
                    cluster_res.H0_high = H0

            # Plot GA image
            fig, ax = plt.subplots(2, 1)
            # Add lines to indicate zones of statistical significant clusters
            for data_name in ['low', 'high']:
                if data_name == 'low':
                    clusters = cluster_res.clusters_low
                    cluster_p_values = cluster_res.cluster_p_values_low
                    ax2 = ax[0]
                elif data_name == 'high':
                    clusters = cluster_res.clusters_high
                    cluster_p_values = cluster_res.cluster_p_values_high
                    ax2 = ax[1]

                for i_c, c in enumerate(clusters):
                    c = c[0]
                    if cluster_p_values[i_c] <= 0.05:
                        h = ax2.axvspan(hf_times[int(tmax*sfreq) + c.start], hf_times[int(tmax*sfreq) + c.stop - 1], color="gray", alpha=0.3)
                        print(data_type)
                        print('Significant Cluster')
                        print(f'p-val: {cluster_p_values[i_c]}')
                        print(hf_times[int(tmax*sfreq) + c.start])
                        print(hf_times[int(tmax*sfreq) + c.stop - 1])

            # Plot actual time courses
            ax[0].plot(epochs_low.times, ga_low_top.reshape(-1), color='limegreen')
            ax[0].fill_between(epochs_low.times, lower_low_top, upper_low_top, color='limegreen', alpha=0.3)
            ax[0].plot(epochs_low.times, ga_low_bottom.reshape(-1), color='darkgreen')
            ax[0].fill_between(epochs_low.times, lower_low_bottom, upper_low_bottom, color='darkgreen',
                               alpha=0.3)
            ax[0].set_title('LF-SEP')
            ax[0].set_ylabel('Amplitude (AU)')
            if not long:
                ax[0].set_xlim([0, 0.07])
            ax[1].plot(hf_times, ga_high_top.reshape(-1), color='limegreen', label='Strongest 200 trials')
            ax[1].fill_between(hf_times, lower_high_top, upper_high_top, color='limegreen', alpha=0.3)
            ax[1].plot(hf_times, ga_high_bottom.reshape(-1), color='darkgreen', label='Weakest 200 trials')
            ax[1].fill_between(hf_times, lower_high_bottom, upper_high_bottom, color='darkgreen', alpha=0.3)
            ax[1].set_title('HFO')
            if not long:
                ax[1].set_xlim([0, 0.07])
            ax[1].set_ylabel('Amplitude (AU)')
            ax[1].set_xlabel('Time (s)')
            plt.legend()
            plt.suptitle(f'GA, {data_type}, {cond_name}, n={len(evoked_list_low_bottom)}')
            plt.tight_layout()
            if not long:
                plt.savefig(figure_path_highlow + f'GA_{cond_name}_env')
                plt.savefig(figure_path_highlow + f'GA_{cond_name}_env.pdf',
                            bbox_inches='tight', format="pdf")
            else:
                plt.savefig(figure_path_highlow + f'GA_{cond_name}_env_long')
                plt.savefig(figure_path_highlow + f'GA_{cond_name}_env_long.pdf',
                            bbox_inches='tight', format="pdf")

            plt.show()
