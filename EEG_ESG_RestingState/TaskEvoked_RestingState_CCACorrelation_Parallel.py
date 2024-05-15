# Want to take 50% of task evoked and resting state trials (resting state trials are dummy)
# Want to compute CCA for both
# Want to store time course of component 1
# Repeat 1000 times
# Get correlation between time courses - should be higher for task evoked versus resting state

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle
import os
import random
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
from EEG_ESG_RestingState.run_CCA_restingstate import run_CCA_restingstate


def get_comp1_evoked(iterable, time_window, epochs_full):
    # print(f'Worker: {iterable}')
    np.random.seed(iterable)
    res = random.sample(range(0, 1999), 1000)
    epochs = epochs_full.copy()[res]
    time_course = run_CCA_restingstate(time_window, epochs)

    return time_course


if __name__ == '__main__':
    freq_band = 'sigma'
    srmr_nr = 2
    iterations = 1000

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
        figure_folder = 'Images'
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'
        figure_folder = 'Images_2'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    timing_path = "/data/pt_02718/Time_Windows.xlsx"  # Contains important info about experiment
    df_timing = pd.read_excel(timing_path)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for data_type in ['spinal', 'subcortical', 'cortical']:
        for condition in conditions:
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            if cond_name in ['median', 'med_mixed']:
                if data_type == 'cortical':
                    window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccacort_med', 'Time'].iloc[0] / 1000,
                                    df_timing.loc[df_timing['Name'] == 'tend_ccacort_med', 'Time'].iloc[0] / 1000]
                elif data_type == 'subcortical':
                    window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccasub_med', 'Time'].iloc[0] / 1000,
                                    df_timing.loc[df_timing['Name'] == 'tend_ccasub_med', 'Time'].iloc[0] / 1000]
                elif data_type == 'spinal':
                    window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccaspinal_med', 'Time'].iloc[0] / 1000,
                                    df_timing.loc[df_timing['Name'] == 'tend_ccaspinal_med', 'Time'].iloc[0] / 1000]
                else:
                    raise RuntimeError('Datatype must be cortical, subcortical or spinal')
            elif cond_name in ['tibial', 'tib_mixed']:
                if data_type == 'cortical':
                    window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccacort_tib', 'Time'].iloc[0] / 1000,
                                    df_timing.loc[df_timing['Name'] == 'tend_ccacort_tib', 'Time'].iloc[0] / 1000]
                elif data_type == 'subcortical':
                    window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccasub_tib', 'Time'].iloc[0] / 1000,
                                    df_timing.loc[df_timing['Name'] == 'tend_ccasub_tib', 'Time'].iloc[0] / 1000]
                elif data_type == 'spinal':
                    window_times = [df_timing.loc[df_timing['Name'] == 'tsart_ccaspinal_tib', 'Time'].iloc[0] / 1000,
                                    df_timing.loc[df_timing['Name'] == 'tend_ccaspinal_tib', 'Time'].iloc[0] / 1000]
                else:
                    raise RuntimeError('Datatype must be cortical, subcortical or spinal')
            else:
                raise RuntimeError('Invalid condition name attempted for use')

            for subject in subjects:
                # Set variables
                subject_id = f'sub-{str(subject).zfill(3)}'
                eeg_chans, spin_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                if cond_name in ['median', 'med_mixed']:
                    esg_chans = cervical_chans
                else:
                    esg_chans = lumbar_chans

                # Select the right files
                if data_type == 'spinal':
                    input_path = f"/data/pt_02718/{folder}/freq_banded_esg/{subject_id}/"
                    save_path = f"/data/pt_02718/{folder}/cca_rs/{subject_id}/"
                elif data_type == 'cortical':
                    input_path = f"/data/pt_02718/{folder}/freq_banded_eeg/{subject_id}/"
                    save_path = f"/data/pt_02718/{folder}/cca_eeg_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    input_path = f"/data/pt_02718/{folder}/freq_banded_eeg/{subject_id}/"
                    save_path = f"/data/pt_02718/{folder}/cca_eeg_thalamic_rs/{subject_id}/"
                figure_path = f"/data/p_02718/{figure_folder}/CCA_RS_Task/{data_type}/"
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(figure_path, exist_ok=True)

                fname_trig = f"{freq_band}_{cond_name}.fif"
                fname_rest = f"{freq_band}_rest_{cond_name}.fif"

                raw_trig = mne.io.read_raw_fif(input_path + fname_trig, preload=True)
                raw_rest = mne.io.read_raw_fif(input_path + fname_rest, preload=True)

                # now create epochs based on the trigger names
                events_trig, event_ids_trig = mne.events_from_annotations(raw_trig)
                event_id_dict_trig = {key: value for key, value in event_ids_trig.items() if key == trigger_name}
                epochs_trig_full = mne.Epochs(raw_trig, events_trig, event_id=event_id_dict_trig, tmin=iv_epoch[0],
                                         tmax=iv_epoch[1] - 1 / 1000,
                                         baseline=tuple(iv_baseline), preload=True, reject_by_annotation=False)

                events_rest, event_ids_rest = mne.events_from_annotations(raw_rest)
                event_id_dict_rest = {key: value for key, value in event_ids_rest.items() if key == trigger_name}
                epochs_rest_full = mne.Epochs(raw_rest, events_rest, event_id=event_id_dict_rest, tmin=iv_epoch[0],
                                         tmax=iv_epoch[1] - 1 / 1000,
                                         baseline=tuple(iv_baseline), preload=True, reject_by_annotation=False)

                if data_type == 'spinal':
                    channels = esg_chans
                elif data_type in ['subcortical', 'cortical']:
                    channels = eeg_chans
                epochs_trig_full = epochs_trig_full.pick_channels(channels, ordered=True)
                epochs_rest_full = epochs_rest_full.pick_channels(channels, ordered=True)

                ind_rest = epochs_rest_full.time_as_index(window_times)
                ind_trig = epochs_trig_full.time_as_index(window_times)

                N = 32  # Too many cores is actually slower
                # N = len(os.sched_getaffinity(0))  # Gives number of available cores
                with mp.Pool(processes=N) as pool:
                    stacked_rest = pool.starmap(get_comp1_evoked, ((iterable, window_times, epochs_rest_full) for iterable in range(iterations)))
                    # Returns list of arrays of size (iterations, time_points)

                with mp.Pool(processes=N) as pool:
                    stacked_trig = pool.starmap(get_comp1_evoked, ((iterable, window_times, epochs_trig_full) for iterable in range(iterations)))

                ############################################################
                # Full epoch correlation
                ############################################################
                R1 = np.corrcoef(stacked_trig)
                fig, ax = plt.subplots(1, 2, figsize=(21, 7))
                c = ax[0].pcolor(abs(R1), vmin=0, vmax=1)
                ax[0].set_title('Task Evoked')
                fig.colorbar(c, ax=ax[0])
                R2 = np.corrcoef(stacked_rest)
                c = ax[1].pcolor(abs(R2), vmin=0, vmax=1)
                ax[1].set_title('Resting State')
                fig.colorbar(c, ax=ax[1])
                plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                plt.savefig(figure_path+f'{subject_id}_{cond_name}')
                plt.close()

                # Save correlation matrices
                afile = open(save_path + f'{data_type}_corr_task_{cond_name}.pkl', 'wb')
                pickle.dump(abs(R1), afile)
                afile.close()

                afile = open(save_path + f'{data_type}_corr_rs_{cond_name}.pkl', 'wb')
                pickle.dump(abs(R2), afile)
                afile.close()

                # Save stacked matrix of time courses
                afile = open(save_path + f'{data_type}_stacked_task_{cond_name}.pkl', 'wb')
                pickle.dump(stacked_trig, afile)
                afile.close()

                afile = open(save_path + f'{data_type}_stacked_rs_{cond_name}.pkl', 'wb')
                pickle.dump(stacked_rest, afile)
                afile.close()
