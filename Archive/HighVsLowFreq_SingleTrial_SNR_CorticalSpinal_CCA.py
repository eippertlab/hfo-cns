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
    data_types = ['Spinal']  # Can be Cortical or Spinal, not both
    # data_types = ['Cortical']

    srmr_nr = 2
    sfreq = 5000
    n_trials = 200 # Number of trials at top/bottom to test
    freq_band = 'sigma'
    perform_pairedttest_and_effsize = True

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
        fig_folder = 'Images'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'
        fig_folder = 'Images_2'

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


    for data_type in data_types:
        df_topbottom10 = pd.DataFrame()
        figure_path_highlow = f'/data/p_02718/{fig_folder}/SingleTrialSNR_LowVsHigh_CCA/StrongestVsWeakest/{data_type}/'
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
                    if cond_name in ['median', 'med_mixed']:
                        expected = 0.013
                        sep_latency = round(df_timing.loc[subject, f"N13"], 3)
                        shift = expected - sep_latency
                    else:
                        expected = 0.022
                        sep_latency = round(df_timing.loc[subject, f"N22"], 3)
                        shift = expected - sep_latency
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
                    df_sub = pd.DataFrame()
                    input_path_snr = f'/data/pt_02718/{folder}/singletrial_snr_cca/{subject_id}/'
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
                    fig, ax = plt.subplots(2, 1)
                    # Not just taking top 200 - need to get indices associated with top and bottom 200 trials first
                    # Also need to be wary the srate of LF files is 1kHz, but HF files are 5kHz
                    hf_times = np.linspace(iv_epoch[0], iv_epoch[1]-1/1000, 1996)
                    ax[0].plot(epochs_low.times, np.mean(epochs_low.get_data(copy=True)[index_high], axis=0).reshape(-1), color='limegreen')
                    ax[0].plot(epochs_low.times, np.mean(epochs_low.get_data(copy=True)[index_low], axis=0).reshape(-1), color='darkgreen')
                    ax[0].set_title('LF-SEP')
                    ax[0].set_ylabel(u'Amplitude (\u03bcV)')
                    ax[0].set_xlim([0, 0.07])
                    ax[1].plot(hf_times, np.mean(epochs.get_data(copy=True)[index_high], axis=0).reshape(-1), color='limegreen', label='Strongest 200 trials')
                    ax[1].plot(hf_times, np.mean(epochs.get_data(copy=True)[index_low], axis=0).reshape(-1), color='darkgreen', label='Weakest 200 trials')
                    ax[1].set_title('HFO')
                    ax[1].set_xlim([0, 0.07])
                    ax[1].set_ylabel('Amplitude (AU)')
                    ax[1].set_xlabel('Time (s)')
                    plt.legend()
                    plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                    plt.tight_layout()
                    plt.savefig(figure_path_highlow+f'{subject_id}_{cond_name}')
                    plt.close()

                    # Add evoked LF-SEP and envelope of HFO to evoked lists to later create GA image
                    # Shift the timing if we're dealing with spinal data
                    if data_type == 'Spinal':
                        evoked_list_low_bottom.append(epochs_low[index_low].shift_time(shift, relative=True).average().data)
                        evoked_list_low_top.append(epochs_low[index_high].shift_time(shift, relative=True).average().data)
                        evoked_list_high_bottom.append(epochs[index_low].shift_time(shift, relative=True).average().apply_hilbert(envelope=True).data)
                        evoked_list_high_top.append(epochs[index_high].shift_time(shift, relative=True).average().apply_hilbert(envelope=True).data)

                    else:
                        evoked_list_low_bottom.append(epochs_low[index_low].average().data)
                        evoked_list_low_top.append(epochs_low[index_high].average().data)
                        evoked_list_high_bottom.append(epochs[index_low].average().apply_hilbert(envelope=True).data)
                        evoked_list_high_top.append(epochs[index_high].average().apply_hilbert(envelope=True).data)

                    # Generate envelope version of plot while we're at it
                    fig, ax = plt.subplots(2, 1)
                    ax[0].plot(epochs_low.times,epochs_low[index_high].average().data.reshape(-1), color='limegreen')
                    ax[0].plot(epochs_low.times,epochs_low[index_low].average().data.reshape(-1),color='darkgreen')
                    ax[0].set_title('LF-SEP')
                    ax[0].set_ylabel(u'Amplitude (\u03bcV)')
                    ax[0].set_xlim([0, 0.07])
                    ax[1].plot(hf_times, epochs[index_high].average().apply_hilbert(envelope=True).data.reshape(-1), color='limegreen', label='Strongest 200 trials')
                    ax[1].plot(hf_times, epochs[index_low].average().apply_hilbert(envelope=True).data.reshape(-1), color='darkgreen', label='Weakest 200 trials')
                    ax[1].set_title('HFO')
                    ax[1].set_xlim([0, 0.07])
                    ax[1].set_ylabel('Amplitude (AU)')
                    ax[1].set_xlabel('Time (s)')
                    plt.legend()
                    plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                    plt.tight_layout()
                    plt.savefig(figure_path_highlow+f'{subject_id}_{cond_name}_env')
                    plt.close()

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
            upper_low_bottom = (ga_low_bottom[0, :] + error_low_bottom).reshape(-1)
            lower_low_bottom = (ga_low_bottom[0, :] - error_low_bottom).reshape(-1)
            error_low_top = sem(evoked_list_low_top, axis=0)
            upper_low_top = (ga_low_top[0, :] + error_low_top).reshape(-1)
            lower_low_top = (ga_low_top[0, :] - error_low_top).reshape(-1)

            ga_high_bottom = np.mean(evoked_list_high_bottom, axis=0)
            ga_high_top = np.mean(evoked_list_high_top, axis=0)
            error_high_bottom = sem(evoked_list_high_bottom, axis=0)
            upper_high_bottom = (ga_high_bottom[0, :] + error_high_bottom).reshape(-1)
            lower_high_bottom = (ga_high_bottom[0, :] - error_high_bottom).reshape(-1)
            error_high_top = sem(evoked_list_high_top, axis=0)
            upper_high_top = (ga_high_top[0, :] + error_high_top).reshape(-1)
            lower_high_top = (ga_high_top[0, :] - error_high_top).reshape(-1)

            # Plot GA image
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(epochs_low.times, ga_low_top.reshape(-1), color='limegreen')
            ax[0].fill_between(epochs_low.times, lower_low_top, upper_low_top, color='limegreen', alpha=0.3)
            ax[0].plot(epochs_low.times, ga_low_bottom.reshape(-1), color='darkgreen')
            ax[0].fill_between(epochs_low.times, lower_low_bottom, upper_low_bottom, color='darkgreen', alpha=0.3)
            ax[0].set_title('LF-SEP')
            ax[0].set_ylabel(u'Amplitude (AU)')
            ax[0].set_xlim([0, 0.07])
            ax[1].plot(hf_times, ga_high_top.reshape(-1), color='limegreen', label='Strongest 200 trials')
            ax[1].fill_between(hf_times, lower_high_top, upper_high_top, color='limegreen', alpha=0.3)
            ax[1].plot(hf_times, ga_high_bottom.reshape(-1), color='darkgreen', label='Weakest 200 trials')
            ax[1].fill_between(hf_times, lower_high_bottom, upper_high_bottom, color='darkgreen', alpha=0.3)
            ax[1].set_title('HFO')
            ax[1].set_xlim([0, 0.07])
            ax[1].set_ylabel('Amplitude (AU)')
            ax[1].set_xlabel('Time (s)')
            plt.legend()
            plt.suptitle(f'GA, {data_type}, {cond_name}, n={len(evoked_list_low_bottom)}')
            plt.tight_layout()
            plt.savefig(figure_path_highlow + f'GA_{cond_name}_env')
            plt.savefig(figure_path_highlow + f'GA_{cond_name}_env.pdf',
                        bbox_inches='tight', format="pdf")
            plt.close()

    # Get percentage change between top and bottom 200 trials
    def percent_change(col1, col2):
        # (top-bottom/bottom)
        return ((col1 - col2) / col2) * 100
    print(df_topbottom10)
    print(df_topbottom10.describe())
    if perform_pairedttest_and_effsize:
        for data_type in data_types:
            for condition in conditions:
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                bottom10_low = df_topbottom10[f'{data_type}_{cond_name}_bottom10_low'].tolist()
                top10_low = df_topbottom10[f'{data_type}_{cond_name}_top10_low'].tolist()
                percent_change_low = percent_change(df_topbottom10[f'{data_type}_{cond_name}_top10_low'], df_topbottom10[f'{data_type}_{cond_name}_bottom10_low'])
                bottom10_high = df_topbottom10[f'{data_type}_{cond_name}_bottom10_high'].tolist()
                top10_high = df_topbottom10[f'{data_type}_{cond_name}_top10_high'].tolist()
                percent_change_high = percent_change(df_topbottom10[f'{data_type}_{cond_name}_top10_high'], df_topbottom10[f'{data_type}_{cond_name}_bottom10_high'])
                print(f'{data_type}, {cond_name}')
                result_low = ttest_rel(bottom10_low, top10_low, nan_policy='omit')
                eff_low = pg.compute_effsize(bottom10_low, top10_low, paired=True, eftype='cohen')
                print('Low')
                print(result_low)
                print(eff_low)
                print(percent_change_low.mean())
                print(percent_change_low.sem())
                result_high = ttest_rel(bottom10_high, top10_high, nan_policy='omit')
                eff_high = pg.compute_effsize(bottom10_high, top10_high, paired=True, eftype='cohen')
                print('High')
                print(result_high)
                print(eff_high)
                print(percent_change_high.mean())
                print(percent_change_high.sem())
                print('\n')
