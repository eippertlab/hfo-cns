# Plot grand average time courses after application of CCA


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.invert import invert
from Common_Functions.GetTimeToAlign_Old import get_time_to_align
import matplotlib.pyplot as plt
import matplotlib as mpl
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


def alignYaxes(axes, align_values=None):
    '''Align the ticks of multiple y axes

    Args:
        axes (list): list of axes objects whose yaxis ticks are to be aligned.
    Keyword Args:
        align_values (None or list/tuple): if not None, should be a list/tuple
            of floats with same length as <axes>. Values in <align_values>
            define where the corresponding axes should be aligned up. E.g.
            [0, 100, -22.5] means the 0 in axes[0], 100 in axes[1] and -22.5
            in axes[2] would be aligned up. If None, align (approximately)
            the lowest ticks in all axes.
    Returns:
        new_ticks (list): a list of new ticks for each axis in <axes>.

        A new sets of ticks are computed for each axis in <axes> but with equal
        length.
    '''
    from matplotlib.pyplot import MaxNLocator

    nax=len(axes)
    ticks=[aii.get_yticks() for aii in axes]
    if align_values is None:
        aligns=[ticks[ii][0] for ii in range(nax)]
    else:
        if len(align_values) != nax:
            raise Exception("Length of <axes> doesn't equal that of <align_values>.")
        aligns=align_values

    bounds=[aii.get_ylim() for aii in axes]

    # align at some points
    ticks_align=[ticks[ii]-aligns[ii] for ii in range(nax)]

    # scale the range to 1-100
    ranges=[tii[-1]-tii[0] for tii in ticks]
    lgs=[-np.log10(rii)+2. for rii in ranges]
    igs=[np.floor(ii) for ii in lgs]
    log_ticks=[ticks_align[ii]*(10.**igs[ii]) for ii in range(nax)]

    # put all axes ticks into a single array, then compute new ticks for all
    comb_ticks=np.concatenate(log_ticks)
    comb_ticks.sort()
    locator=MaxNLocator(nbins='auto', steps=[1, 2, 2.5, 3, 4, 5, 8, 10])
    new_ticks=locator.tick_values(comb_ticks[0], comb_ticks[-1])
    new_ticks=[new_ticks/10.**igs[ii] for ii in range(nax)]
    new_ticks=[new_ticks[ii]+aligns[ii] for ii in range(nax)]

    # find the lower bound
    idx_l=0
    for i in range(len(new_ticks[0])):
        if any([new_ticks[jj][i] > bounds[jj][0] for jj in range(nax)]):
            idx_l=i-1
            break

    # find the upper bound
    idx_r=0
    for i in range(len(new_ticks[0])):
        if all([new_ticks[jj][i] > bounds[jj][1] for jj in range(nax)]):
            idx_r=i
            break

    # trim tick lists by bounds
    new_ticks=[tii[idx_l:idx_r+1] for tii in new_ticks]

    # set ticks for each axis
    for axii, tii in zip(axes, new_ticks):
        axii.set_yticks(tii)

    return new_ticks


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_bands = ['sigma']
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    # Get a raw file so I can use the montage
    raw = mne.io.read_raw_fif("/data/pt_02718/tmp_data/freq_banded_eeg/sub-001/sigma_median.fif", preload=True)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)
    raw.set_montage(montage, on_missing="ignore")
    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
    res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)

    figure_path = '/data/p_02718/SAB/GrandAverage_AverageLat_YY/'
    os.makedirs(figure_path, exist_ok=True)

    # Cortical Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
    df_vis_cortical = pd.read_excel(xls, 'CCA_Brain')
    df_vis_cortical.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG.xlsx')
    df_cortical = pd.read_excel(xls, 'CCA')
    df_cortical.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Cortical_Timing.xlsx')
    df_timing = pd.read_excel(xls, 'Timing')
    df_timing.set_index('Subject', inplace=True)

    # Spinal Excel files
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
    df_spinal = pd.read_excel(xls, 'CCA_goodonly')
    df_spinal.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
    df_vis_spinal = pd.read_excel(xls, 'CCA_Spinal_GoodOnly')
    df_vis_spinal.set_index('Subject', inplace=True)

    use_visible = True  # Use only subjects with visible bursting

    for data_type in ['eeg', 'esg']:
        for freq_band in freq_bands:
            for condition in conditions:
                evoked_list = []
                spatial_pattern = []
                evoked_list_low = []

                if condition == 2:
                    if data_type == 'eeg':
                        target_electrode = 'CP4'
                    else:
                        target_electrode = 'SC6'
                elif condition == 3:
                    if data_type == 'eeg':
                        target_electrode = 'Cz'
                    else:
                        target_electrode = 'L1'

                for subject in subjects:
                    # Set variables
                    cond_info = get_conditioninfo(condition, srmr_nr)
                    cond_name = cond_info.cond_name
                    trigger_name = cond_info.trigger_name
                    subject_id = f'sub-{str(subject).zfill(3)}'

                    ##########################################################
                    # Time  Course Information
                    ##########################################################
                    # Select the right files
                    if data_type == 'eeg':
                        color = 'tab:purple'
                        color_low = 'tab:red'
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                        df = df_cortical
                        df_vis = df_vis_cortical

                        # Low Freq SEP
                        input_path_low = "/data/pt_02068/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                        fname_low = f"cnt_clean_{cond_name}.set"
                        raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)
                        raw.set_montage(montage, on_missing="ignore")
                        evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)

                    else:
                        color = 'tab:blue'
                        color_low = 'deeppink'
                        # HFO
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca_goodonly/" + subject_id + "/"
                        df = df_spinal
                        df_vis = df_vis_spinal

                        potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
                        fname_pot = 'potential_latency.mat'
                        matdata = loadmat(potential_path + fname_pot)

                        # Low Freq SEP
                        input_path_low = f"/data/p_02569/SSP/{subject_id}/6 projections/"
                        fname_low = f"epochs_{cond_name}.fif"
                        epochs_low = mne.read_epochs(input_path_low + fname_low, preload=True)
                        evoked_low = epochs_low.average()

                    epochs = mne.read_epochs(input_path + fname, preload=True)

                    # Need to pick channel based on excel sheet
                    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    channel = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()

                    # Apply relative time-shift depending on expected latency
                    if data_type == 'eeg':
                        median_lat, tibial_lat = get_time_to_align('eeg', ['median', 'tibial'], np.arange(1, 37))
                        if cond_name == 'median':
                            sep_latency = df_timing.loc[subject, f"N20"]
                            expected = median_lat
                        elif cond_name == 'tibial':
                            sep_latency = df_timing.loc[subject, f"P39"]
                            expected = tibial_lat
                        shift = sep_latency - expected
                        evoked.shift_time(shift, relative=True)
                        evoked_low.shift_time(shift, relative=True)

                    elif data_type == 'esg':
                        median_lat, tibial_lat = get_time_to_align('esg', ['median', 'tibial'], np.arange(1, 37))
                        if cond_name == 'median':
                            sep_latency = matdata['med_potlatency']
                            expected = median_lat
                        elif cond_name == 'tibial':
                            sep_latency = matdata['tib_potlatency']
                            expected = tibial_lat
                        shift = sep_latency[0][0] / 1000 - expected
                        evoked.shift_time(shift, relative=True)
                        evoked_low.shift_time(shift, relative=True)

                    evoked.crop(tmin=-0.06, tmax=0.07)
                    evoked_low.crop(tmin=-0.06, tmax=0.07)
                    data = evoked.data

                    if use_visible is True:
                        visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                        if visible == 'T':
                            evoked_list.append(data)
                            evoked_list_low.append(evoked_low.get_data(picks=[target_electrode]).reshape(-1))
                    else:
                        evoked_list.append(data)
                        evoked_list_low.append(evoked_low.get_data(picks=[target_electrode]).reshape(-1))

                # Get grand average across chosen epochs
                grand_average = np.mean(evoked_list, axis=0)
                grand_average_low = np.mean(evoked_list_low, axis=0)

                # Plot Time Course as YY plot
                fig = plt.figure()
                ax1 = plt.subplot(211)
                ax10 = plt.subplot(212)
                # ax10 = ax1.twinx()

                # HFO
                ax1.plot(evoked.times, grand_average[0, :], color=color)
                ax1.set_ylabel('HFO Amplitude (AU)')
                ax1.set_xticks([])
                ax1.spines['bottom'].set_visible(False)

                ax1.set_title(f'Grand Average Time Courses, n={len(evoked_list)}')
                ax1.set_xlim([0.00, 0.07])
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)

                # Low Freq
                ax10.plot(evoked_low.times, grand_average_low * 10 ** 6, label='SSP6',
                          color=color_low)
                # ax3.set_yticklabels([])
                ax10.set_ylabel('SEP Amplitude (\u03BCV)')
                # ax10.spines['left'].set_color(color_low)
                ax10.spines['top'].set_visible(False)
                ax10.spines['right'].set_visible(False)
                # ax1.spines['left'].set_color(color)
                ax10.set_xlabel('Time (s)')
                # ax1.tick_params(axis='y', colors=color)
                # ax10.tick_params(axis='y', colors=color_low)
                ax10.set_xlim([0.00, 0.07])

                # align ax10 ylabel to ax1 ylabel
                fig.align_ylabels([ax1, ax10])

                # Align so y axis of SEP is below other
                if data_type == 'eeg':
                    if condition == 2:
                        ax1.set_ylim([-0.2, 0.2])
                    elif condition == 3:
                        ax1.set_ylim([-0.04, 0.04])
                elif data_type == 'esg':
                    ax1.set_ylim([-0.08, 0.08])
                    # alignYaxes([ax1, ax10], [0, 0.5])

                # alignYaxes([ax1, ax10], [0, 4])

                if use_visible is True:
                    plt.tight_layout()
                    plt.savefig(figure_path+f'{data_type}_GA_Time_{freq_band}_{cond_name}_visible')
                    plt.savefig(figure_path+f'{data_type}_GA_Time_{freq_band}_{cond_name}_visible.pdf',
                                bbox_inches='tight', format="pdf")
                else:
                    plt.tight_layout()
                    plt.savefig(figure_path + f'{data_type}_GA_Time_{freq_band}_{cond_name}')
                    plt.savefig(figure_path + f'{data_type}_GA_Time_{freq_band}_{cond_name}.pdf', bbox_inches='tight',
                                format="pdf")