# IR = [(sum(D1, D2) - D1D2) / sum(D1, D2)]  * 100
# Want to look at interaction ratio for each subject in high and low frequency condition
# IR captures amplitude attenuation caused by simultaneous stimulation of two digits
# Taking AMPLITUDE NOT MAGNITUDE
# Sign has meaning here
# Use channel CP4 for low_freq, correct CCA component for high freq
# Can do one-sample t-tests with IR values against 0 to determine statistical significance

import mne
import os
import numpy as np
from Common_Functions.envelope_noise_reduction import envelope_noise_reduction
from scipy.interpolate import PchipInterpolator as pchip
from Common_Functions.invert import invert
from Common_Functions.get_conditioninfo import get_conditioninfo
import pingouin as pg
from Common_Functions.calculate_snr_hfo import calculate_snr
from Common_Functions.calculate_snr_lowfreq import calculate_snr_lowfreq
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from scipy.stats import pearsonr
from Common_Functions.check_excel_exist_partialcorr import check_excel_exist_partialcorr
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    #######################################################
    # Set paths and variables
    #######################################################
    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df_det = pd.read_excel(cfg_path)
    iv_baseline = [df_det.loc[df_det['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df_det.loc[df_det['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df_det.loc[df_det['var_name'] == 'epoch_start', 'var_value'].iloc[0],
                df_det.loc[df_det['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    srmr_nr = 2
    if srmr_nr != 2:
        raise RuntimeError('Experiment number must be 2 here')
    sfreq = 5000
    freq_band = 'sigma'

    subjects = np.arange(1, 25)
    condition_dig = 2  # Interested in med_digits case here, want to normalise by wrist activity
    cond_info = get_conditioninfo(condition_dig, srmr_nr)
    cond_name_dig = cond_info.cond_name
    trigger_names = cond_info.trigger_name

    condition_wrist = 3  # Interested in med_digits case here, want to normalise by wrist activity
    cond_info_wrist = get_conditioninfo(condition_wrist, srmr_nr)
    cond_name_wrist = cond_info_wrist.cond_name
    trigger_name_wrist = cond_info_wrist.trigger_name

    time_edge = 0.004
    time_peak = 0.022
    time_peak_wrist = 0.02
    channel_low = ['CP4']

    figure_path = '/data/p_02718/Images_2/CCA_eeg_digits/InteractionRatio_Normalised/'
    os.makedirs(figure_path, exist_ok=True)

    # Cortical Excel files and image path - digits data
    xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated_Digits.xlsx')
    df_dig = pd.read_excel(xls, 'CCA')
    df_dig.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated_Digits.xlsx')
    df_vis_dig = pd.read_excel(xls, 'CCA_Brain')
    df_vis_dig.set_index('Subject', inplace=True)

    # Cortical Excel files and image path - wrist data
    xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated.xlsx')
    df_wrist = pd.read_excel(xls, 'CCA')
    df_wrist.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated.xlsx')
    df_vis_wrist = pd.read_excel(xls, 'CCA_Brain')
    df_vis_wrist.set_index('Subject', inplace=True)

    IR_low_list = []
    IR_high_list = []
    subj_list = []
    med1_amplow_list = []
    med2_amplow_list = []
    med12_amplow_list = []
    med1_amphigh_list = []
    med2_amphigh_list = []
    med12_amphigh_list = []

    for subject in subjects:
        subject_id = f'sub-{str(subject).zfill(3)}'

        #######################################################################################################
        # High Frequency Amplitude Envelope
        #######################################################################################################
        input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"

        # WRIST
        # Get wrist peak amplitude of envelope so we know what to normalise by
        channel_no_wrist = df_wrist.loc[subject, f"{freq_band}_{cond_name_wrist}_comp"]
        fname = f"{freq_band}_{cond_name_wrist}.fif"
        epochs_wrist = mne.read_epochs(input_path + fname, preload=True)
        channel = f'Cor{int(channel_no_wrist)}'
        inv = df_wrist.loc[subject, f"{freq_band}_{cond_name_wrist}_flip"]
        if inv == 'T':
            epochs_wrist.apply_function(invert, picks=channel)
        evoked_wrist = epochs_wrist.copy().average()
        envelope_wrist = evoked_wrist.apply_hilbert(envelope=True)
        data_wrist = envelope_wrist.copy().crop(tmin=time_peak_wrist - time_edge, tmax=time_peak_wrist + time_edge).get_data()
        # Want to subtract mean noise in noise period (-100ms to -10ms) from each data point in the envelope
        noise_data_wrist = envelope_wrist.copy().crop(-100 / 1000, -10 / 1000).get_data()
        cleaned_data_wrist = envelope_noise_reduction(data_wrist, noise_data_wrist)
        amplitude_high_wrist = np.max(cleaned_data_wrist[0])

        # DIGITS
        channel_no_dig = df_dig.loc[subject, f"{freq_band}_{cond_name_dig}_comp"]
        visible_dig = df_vis_dig.loc[subject, f"{freq_band.capitalize()}_{cond_name_dig.capitalize()}_Visible"]

        if visible_dig == 'T':  # Only perform for retained subjects
            subj_list.append(subject)
            fname = f"{freq_band}_{cond_name_dig}.fif"
            epochs_high_all = mne.read_epochs(input_path + fname, preload=True)
            channel = f'Cor{int(channel_no_dig)}'
            inv = df_dig.loc[subject, f"{freq_band}_{cond_name_dig}_flip"]
            epochs_high_all = epochs_high_all.pick_channels([channel])

            for trigger_name, amp_high_list in zip(trigger_names,
                                                   [med1_amphigh_list, med2_amphigh_list, med12_amphigh_list]):
                epochs = epochs_high_all[trigger_name]

                # Need to pick channel based on excel sheet
                channel = f'Cor{int(channel_no_dig)}'
                inv = df_dig.loc[subject, f"{freq_band}_{cond_name_dig}_flip"]
                epochs = epochs.pick_channels([channel])
                if inv == 'T':
                    epochs.apply_function(invert, picks=channel)
                evoked = epochs.copy().average()
                envelope = evoked.apply_hilbert(envelope=True)
                data = envelope.copy().crop(tmin=time_peak - time_edge, tmax=time_peak + time_edge).get_data()

                # Want to subtract mean noise in noise period (-100ms to -10ms) from each data point in the envelope
                noise_data = envelope.copy().crop(-100 / 1000, -10 / 1000).get_data()
                cleaned_data = envelope_noise_reduction(data, noise_data)
                amplitude_high = np.max(cleaned_data[0])
                amp_high_list.append(amplitude_high/amplitude_high_wrist)

            ################################################################################################
            # Low Frequency Amplitude
            ################################################################################################
            input_path_low = "/data/pt_02151/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"

            # WRIST
            fname_low_wrist = f"cnt_clean_{cond_name_wrist}.set"
            raw = mne.io.read_raw_eeglab(input_path_low + fname_low_wrist, preload=True)
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_name_wrist}
            epochs_low = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                    baseline=tuple(iv_baseline))
            evoked_low_wrist = epochs_low.copy().average()
            evoked_low_wrist.crop(tmin=-0.06, tmax=0.07)
            data_low_wrist = evoked_low_wrist.crop(tmin=time_peak_wrist - time_edge, tmax=time_peak_wrist + time_edge).get_data().reshape(
                -1)
            if min(data_low_wrist) < 0:
                _, latency_low_wrist, amplitude_low_wrist = evoked_low_wrist.get_peak(tmin=time_peak_wrist - time_edge,
                                                                                      tmax=time_peak_wrist + time_edge,
                                                                                      mode='neg', return_amplitude=True)
            else:
                latency_low_wrist = time_peak_wrist
                amplitude_low_wrist = np.nan

            # DIGITS
            fname_low = f"cnt_clean_{cond_name_dig}.set"
            raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key in trigger_names}
            epochs_low_all = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                        baseline=tuple(iv_baseline))

            for trigger_name, amp_low_list in zip(trigger_names,
                                                   [med1_amplow_list, med2_amplow_list, med12_amplow_list]):
                epochs_low = epochs_low_all[trigger_name]
                evoked_low = epochs_low.copy().average()
                evoked_low.crop(tmin=-0.06, tmax=0.07)
                data_low = evoked_low.crop(tmin=time_peak - time_edge, tmax=time_peak + time_edge).get_data().reshape(
                    -1)
                if min(data_low) < 0:
                    _, latency_low, amplitude_low = evoked_low.get_peak(tmin=time_peak - time_edge,
                                                                        tmax=time_peak + time_edge,
                                                                        mode='neg', return_amplitude=True)
                else:
                    latency_low = time_peak
                    amplitude_low = np.nan
                amp_low_list.append(amplitude_low/amplitude_low_wrist)

    print('Low Amplitudes')
    for amp_low_list in [med1_amplow_list, med2_amplow_list, med12_amplow_list]:
        print(amp_low_list)

    print('High Amplitudes')
    for amp_high_list in [med1_amphigh_list, med2_amphigh_list, med12_amphigh_list]:
        print(amp_high_list)

    ################################################################################
    # Interaction Ratio per Subject
    ################################################################################
    # IR = [(sum(D1, D2) - D1D2) / sum(D1, D2)]  * 100
    print(subj_list)
    for i in np.arange(0, len(subj_list)):
        i -= 1  # So we can index with it

        ir_low = (((med1_amplow_list[i]+med2_amplow_list[i]) - med12_amplow_list[i]) / (med1_amplow_list[i]+med2_amplow_list[i]))*100
        ir_high = (((med1_amphigh_list[i]+med2_amphigh_list[i]) - med12_amphigh_list[i]) / (med1_amphigh_list[i]+med2_amphigh_list[i]))*100

        IR_low_list.append(ir_low)
        IR_high_list.append(ir_high)

    # print('IR Low')
    # print(IR_low_list)
    # print(np.mean(IR_low_list))
    #
    # print('IR High')
    # print(IR_high_list)
    # print(np.mean(IR_high_list))

    df = pd.DataFrame()
    df['Subject'] = subj_list
    df['Low Frequency'] = IR_low_list
    df['High Frequency'] = IR_high_list
    print(df)
    df_longform = pd.melt(df, id_vars=['Subject'],
                          value_vars=[f'Low Frequency', f'High Frequency'],
                          var_name='Frequency Level', value_name='Interaction Ratio')  # Change to long format
    # print(df_longform)

    plt.figure()
    g = sns.catplot(kind='point', data=df_longform, x='Frequency Level', y='Interaction Ratio', hue='Subject', color='gray')
    for ax in g.axes.flat:
        for line in ax.lines:
            line.set_alpha(0.3)
        for dots in ax.collections:
            color = dots.get_facecolor()
            dots.set_color(sns.set_hls_values(color, l=0.5))
            dots.set_alpha(0.3)
    g.map_dataframe(sns.boxplot, data=df_longform, x='Frequency Level', y='Interaction Ratio', hue='Frequency Level',
                    dodge=False, palette=['tab:red', 'tab:purple'])
    g.fig.set_size_inches(16, 10)
    g._legend.remove()
    plt.ylabel('IR (%)')
    plt.savefig(figure_path+f'Boxplot_IR_HighvLow_{cond_name_dig}')
    # plt.show()

    plt.figure()
    # Scatter plot with correlation coeff
    sns.scatterplot(data=df, x='Low Frequency', y='High Frequency')
    # pearson_corr_og = df.abs()[f'{col_names[0]}'].corr(df.abs()[f'{col_names[1]}'])
    df.dropna(inplace=True)
    pearson_corr = pearsonr(df[f'Low Frequency'], df[f'High Frequency'])
    # g.fig.set_size_inches(16, 10)
    plt.title(f"PearsonCorrelation: {round(pearson_corr.statistic, 4)}, pval: {round(pearson_corr.pvalue, 4)}")
    plt.savefig(figure_path+f'Scatterplot_IR_HighvLow_{cond_name_dig}')

    # Quick ttest
    stats_high = pg.ttest(x=IR_high_list, y=0)
    stats_low = pg.ttest(x=IR_low_list, y=0)
    stats_diff = pg.ttest(x=IR_low_list, y=IR_high_list, paired=True)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 150)
    print(stats_high)
    print(stats_low)
    print(stats_diff)

    plt.show()
