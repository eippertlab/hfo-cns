# Script to plot the time-frequency decomposition about the spinal triggers for the correct versus incorrect patch
# Also include the cortical data for just the correct cluster
# https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html#morlet-wavelets

import mne
import os
import numpy as np
import pickle
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.appy_cca_weights import apply_cca_weights
from Common_Functions.get_channels import get_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':

    shift_spinal = False  # If true, shift the spinal based on time of underlying low freq SEP

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    for srmr_nr in [1, 2]:
        if srmr_nr == 1:
            image_path = "/data/p_02718/Polished/TFRs_SingleChannel_CCAtoBroadband/"
            os.makedirs(image_path, exist_ok=True)

            image_path_ss = "/data/p_02718/Polished/TFRs_SingleChannel_CCAtoBroadband/SingleSubjects/"
            os.makedirs(image_path_ss, exist_ok=True)

            folder = 'tmp_data'
            subjects = np.arange(1, 37)
            sfreq = 5000
            conditions = [2, 3]

        elif srmr_nr == 2:
            image_path = "/data/p_02718/Polished_2/TFRs_SingleChannel_CCAtoBroadband/"
            os.makedirs(image_path, exist_ok=True)

            image_path_ss = "/data/p_02718/Polished_2/TFRs_SingleChannel_CCAtoBroadband/SingleSubjects/"
            os.makedirs(image_path_ss, exist_ok=True)

            folder = 'tmp_data_2'
            subjects = np.arange(1, 25)
            sfreq = 5000
            conditions = [3, 5]

        eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
        brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

        freq_band = 'sigma'

        # Spinal Timing Excel File
        xls_timing = pd.ExcelFile(f'/data/pt_02718/{folder}/LowFreq_HighFreq_Relation.xlsx')
        df_timing_spinal = pd.read_excel(xls_timing, 'Spinal')
        df_timing_spinal.set_index('Subject', inplace=True)

        # Cortical Excel files
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Updated.xlsx')
        df_cortical = pd.read_excel(xls, 'CCA')
        df_cortical.set_index('Subject', inplace=True)

        # Thalamic Excel files
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_EEG_Thalamic_Updated.xlsx')
        df_thalamic = pd.read_excel(xls, 'CCA')
        df_thalamic.set_index('Subject', inplace=True)

        # Spinal Excel files
        xls = pd.ExcelFile(f'/data/pt_02718/{folder}/Components_Updated.xlsx')
        df_spinal = pd.read_excel(xls, 'CCA')
        df_spinal.set_index('Subject', inplace=True)

        for freq_type in ['upper', 'full']:
            if freq_type == 'full':
                freqs = np.arange(0., 1200., 3.)
                fmin, fmax = freqs[[0, -1]]
            elif freq_type == 'upper':
                freqs = np.arange(200., 1200., 3.)
                fmin, fmax = freqs[[0, -1]]

            # To use mne grand_average method, need to generate a list of evoked potentials for each subject
            for condition in conditions:
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                evoked_list_cortical = []
                evoked_list_thalamic = []
                evoked_list_spinal = []

                for data_type, evoked_list in zip(['Spinal', 'Thalamic', 'Cortical'],
                                                  [evoked_list_spinal, evoked_list_thalamic, evoked_list_cortical]):
                    for subject in subjects:  # All subjects
                        subject_id = f'sub-{str(subject).zfill(3)}'

                        if data_type == 'Spinal':
                            df = df_spinal
                            with open(f'/data/pt_02718/{folder}/cca/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                                      'rb') as f:
                                W_st = pickle.load(f)
                        elif data_type == 'Thalamic':
                            df = df_thalamic
                            with open(
                                    f'/data/pt_02718/{folder}/cca_eeg_thalamic/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                                    'rb') as f:
                                W_st = pickle.load(f)
                        elif data_type == 'Cortical':
                            df = df_cortical
                            with open(f'/data/pt_02718/{folder}/cca_eeg/{subject_id}/W_st_{freq_band}_{cond_name}.pkl',
                                      'rb') as f:
                                W_st = pickle.load(f)
                        else:
                            raise RuntimeError('This given datatype is not one of Spinal/Thalamic/Cortical')

                        # Read in data
                        if data_type in ['Cortical', 'Thalamic']:
                            fname = f"noStimart_sr{sfreq}_{cond_name}_withqrs_eeg.fif"
                            input_path = f"/data/pt_02718/{folder}/imported/{subject_id}/"

                        elif data_type == 'Spinal':
                            fname = f"ssp6_cleaned_{cond_name}.fif"
                            input_path = f"/data/pt_02718/{folder}/ssp_cleaned/{subject_id}/"

                        raw_data = mne.io.read_raw_fif(input_path + fname, preload=True)

                        events, event_ids = mne.events_from_annotations(raw_data)
                        event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
                        epochs = mne.Epochs(raw_data, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                                            baseline=tuple(iv_baseline), preload=True)

                        apply_weights_kwargs = dict(
                            weights=W_st
                        )
                        cor_names = [f'Cor{i}' for i in np.arange(1, len(W_st) + 1)]
                        if data_type == 'Spinal':
                            if cond_name in ['median', 'med_mixed']:
                                epochs.pick(cervical_chans).reorder_channels(cervical_chans)
                                epochs_ccafiltered = epochs.apply_function(apply_cca_weights, channel_wise=False,
                                                                           **apply_weights_kwargs)
                                # Remap names to match Cor 1, Cor2 etc
                                channel_map = {cervical_chans[i]: cor_names[i] for i in range(len(cervical_chans))}
                                epochs_ccafiltered.rename_channels(channel_map)
                            elif cond_name in ['tibial', 'tib_mixed']:
                                epochs.pick(lumbar_chans).reorder_channels(lumbar_chans)
                                epochs_ccafiltered = epochs.apply_function(apply_cca_weights, channel_wise=False,
                                                                           **apply_weights_kwargs)
                                channel_map = {lumbar_chans[i]: cor_names[i] for i in range(len(lumbar_chans))}
                                epochs_ccafiltered.rename_channels(channel_map)
                            evoked_ccafiltered = epochs_ccafiltered.average()
                        elif data_type in ['Thalamic', 'Cortical']:
                            epochs.pick(eeg_chans).reorder_channels(eeg_chans)
                            epochs_ccafiltered = epochs.apply_function(apply_cca_weights, picks=eeg_chans,
                                                                       channel_wise=False, **apply_weights_kwargs)
                            channel_map = {eeg_chans[i]: cor_names[i] for i in range(len(eeg_chans))}
                            epochs_ccafiltered.rename_channels(channel_map)
                            evoked_ccafiltered = epochs_ccafiltered.average()

                        if shift_spinal and data_type == 'Spinal':
                            # Apply relative time-shift depending on expected latency for spinal data
                            # median_lat, tibial_lat = get_time_to_align('esg', ['median', 'tibial'], np.arange(1, 37))
                            median_lat = 0.013
                            tibial_lat = 0.022
                            if cond_name in ['median', 'med_mixed']:
                                sep_latency = round(df_timing_spinal.loc[subject, f"N13"], 3)
                                expected = median_lat
                            elif cond_name in ['tibial', 'tib_mixed']:
                                sep_latency = round(df_timing_spinal.loc[subject, f"N22"], 3)
                                expected = tibial_lat
                            shift = expected - sep_latency
                            evoked_ccafiltered.shift_time(shift, relative=True)

                        evoked_ccafiltered.crop(tmin=-0.06, tmax=0.1)
                        channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                        channel = f'Cor{channel_no}'
                        inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]

                        if channel_no != 0:  # 0 marks subjects where no component is selected
                            evoked_ccafiltered.pick(channel)
                            # Renaming them all to Cor1 so we can do grand average - since we have already picked the right
                            # channel, we're just renaming it
                            evoked_ccafiltered.rename_channels({channel:'Cor1'}, allow_duplicates=False, verbose=None)
                            # Get power
                            power = mne.time_frequency.tfr_stockwell(evoked_ccafiltered, fmin=fmin, fmax=fmax, width=3.0,
                                                                              n_jobs=5)

                            fig, ax = plt.subplots(1, 1, figsize=[6, 10])
                            tmin = 0.0
                            tmax = 0.06
                            # Because combine = 'mean', the data in all channels is averaged as picks = 'eeg'
                            power.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                                   axes=ax, show=False, colorbar=True, dB=False,
                                                   tmin=tmin, tmax=tmax, vmin=0)
                            im = ax.images
                            cb = im[-1].colorbar
                            cb.set_label('Amplitude (AU)')

                            ax.set_title(f"{subject_id}, {data_type}, {cond_name}, CCAtoBroadband")
                            if freq_type == 'full':
                                fname = f"{subject_id}_{data_type}_{trigger_name}_full_ratio"
                            elif freq_type == 'upper':
                                fname = f"{subject_id}_{data_type}_{trigger_name}_ratio"
                            if shift_spinal:
                                fig.savefig(image_path_ss + fname + '_spinalshifted_longcrop.png')
                                # plt.savefig(image_path + fname + '_spinalshifted_longcrop.pdf', bbox_inches='tight',
                                #             format="pdf")
                            else:
                                fig.savefig(image_path_ss + fname + '_longcrop.png')
                                # plt.savefig(image_path + fname + '_longcrop.pdf', bbox_inches='tight', format="pdf")
                            plt.clf()

                            evoked_list.append(power)

                # Get grand average across subjects
                averaged_cortical = mne.grand_average(evoked_list_cortical, interpolate_bads=False, drop_bads=False)
                averaged_thalamic = mne.grand_average(evoked_list_thalamic, interpolate_bads=False, drop_bads=False)
                averaged_spinal = mne.grand_average(evoked_list_spinal, interpolate_bads=False, drop_bads=False)

                fig, ax = plt.subplots(1, 3, figsize=[18, 6])
                ax = ax.flatten()
                tmin = 0.0
                tmax = 0.06
                if cond_name in ['median', 'med_mixed']:
                    vmax_cortical = 260
                    vmax_thalamic = 120
                    vmax_spinal = 150
                    vmin = 0
                elif cond_name in ['tibial', 'tib_mixed']:
                    vmax_cortical = 35
                    vmax_thalamic = 20
                    vmax_spinal = 15
                    vmin = 0
                # Because combine = 'mean', the data in all channels is averaged as picks = 'eeg'
                averaged_cortical.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                      axes=ax[0], show=False, colorbar=True, dB=False,
                                      tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax_cortical, combine='mean')
                averaged_thalamic.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                      axes=ax[1], show=False, colorbar=True, dB=False,
                                      tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax_thalamic, combine='mean')
                averaged_spinal.plot(picks='eeg', baseline=iv_baseline, mode='ratio', cmap='jet',
                                        axes=ax[2], show=False, colorbar=True, dB=False,
                                        tmin=tmin, tmax=tmax, vmin=vmin, vmax=vmax_spinal, combine='mean')
                for axes in [ax[0], ax[1], ax[2]]:
                    im = axes.images
                    cb = im[-1].colorbar
                    cb.set_label('Amplitude (AU)')

                ax[0].set_title(f"Grand average cortical")
                ax[1].set_title(f"Grand average thalamic")
                ax[2].set_title(f"Grand average spinal")
                if freq_type == 'full':
                    fname = f"{trigger_name}_full_ratio"
                elif freq_type == 'upper':
                    fname = f"{trigger_name}_ratio"
                plt.tight_layout()
                if shift_spinal:
                    fig.savefig(image_path + fname + '_spinalshifted_longcrop.png')
                    plt.savefig(image_path + fname+'_spinalshifted_longcrop.pdf', bbox_inches='tight', format="pdf")
                else:
                    fig.savefig(image_path + fname+'_longcrop.png')
                    plt.savefig(image_path + fname+'_longcrop.pdf', bbox_inches='tight', format="pdf")
                plt.clf()
