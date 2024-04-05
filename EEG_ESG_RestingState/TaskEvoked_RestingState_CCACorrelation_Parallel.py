# Want to take 50% of task evoked and resting state trials (resting state trials are dummy)
# Want to compute CCA for both
# Want to store time course of component 1
# Repeat 1000 times
# Get correlation between time courses - should be higher for task evoked versus resting state

import numpy as np
import pandas as pd
import mne
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import repeat
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import random
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
from EEG_ESG_RestingState.run_CCA_restingstate import run_CCA_restingstate


def get_comp1_evoked(condition_name, dtype, epochs_full):
    # For come reason once I get in here, epochs_full is no longer mne.Epochs class, but rather np.array
    print(epochs_full)
    print(dtype)
    print(condition_name)
    stacked = np.empty((iterations, 1996))  # Iterations by timepoints
    for iteration in np.arange(0, iterations):
        res = random.sample(range(0, 1999), 1000)
        # Select relevant channels and trials
        if dtype == 'spinal':
            epochs = epochs_full.copy().pick_channels(esg_chans, ordered=True)[res]
        elif dtype in ['subcortical', 'cortical']:
            epochs = epochs_full.copy().pick_channels(eeg_chans, ordered=True)[res]

        stacked[iteration, :] = run_CCA_restingstate(condition_name, data_type, epochs)

    return stacked


if __name__ == '__main__':
    freq_band = 'sigma'
    srmr_nr = 1
    iterations = 1000

    if srmr_nr == 1:
        subjects = np.arange(1, 11)  # Haven't generated all RS data yet
        conditions = [2, 3]
    else:
        raise RuntimeError('Only implemented for srmr_nr 1')

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for data_type in ['spinal', 'subcortical', 'cortical']:
        for condition in conditions:
            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'
                eeg_chans, spin_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)
                if cond_name in ['median', 'med_mixed']:
                    esg_chans = cervical_chans
                else:
                    esg_chans = lumbar_chans

                # Select the right files
                if data_type == 'spinal':
                    input_path = f"/data/pt_02718/tmp_data/freq_banded_esg/{subject_id}/"
                    save_path = f"/data/pt_02718/tmp_data/cca_rs/{subject_id}/"
                elif data_type == 'cortical':
                    input_path = f"/data/pt_02718/tmp_data/freq_banded_eeg/{subject_id}/"
                    save_path = f"/data/pt_02718/tmp_data/cca_eeg_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    input_path = f"/data/pt_02718/tmp_data/freq_banded_eeg/{subject_id}/"
                    save_path = f"/data/pt_02718/tmp_data/cca_eeg_thalamic_rs/{subject_id}/"
                figure_path = f"/data/p_02718/Images/CCA_RS_Task/{data_type}/"
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

                print(type(epochs_rest_full))
                print(type(epochs_trig_full))
                N = mp.cpu_count()
                with mp.Pool(processes=N) as pool:
                    print(epochs_rest_full)
                    stacked_rest = pool.starmap(get_comp1_evoked, zip(repeat(cond_name), repeat(data_type), epochs_rest_full))

                with mp.Pool(processes=N) as pool:
                    stacked_trig = pool.starmap(get_comp1_evoked, zip(repeat(cond_name), repeat(data_type), epochs_trig_full))

                R1 = np.corrcoef(stacked_trig)
                fig, ax = plt.subplots(1, 2, figsize=(21, 7))
                c = ax[0].pcolor(R1, vmin=-1, vmax=1)
                ax[0].set_title('Task Evoked')
                fig.colorbar(c, ax=ax[0])
                R2 = np.corrcoef(stacked_rest)
                c = ax[1].pcolor(R2, vmin=-1, vmax=1)
                ax[1].set_title('Resting State')
                fig.colorbar(c, ax=ax[1])
                plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                plt.savefig(figure_path+f'{subject_id}_{cond_name}_parallel')
                plt.close()

                # # Save correlation matrices and image
                # afile = open(save_path + f'{data_type}_corr_task_{cond_name}.pkl', 'wb')
                # pickle.dump(R1, afile)
                # afile.close()
                #
                # afile = open(save_path + f'{data_type}_corr_rs_{cond_name}.pkl', 'wb')
                # pickle.dump(R2, afile)
                # afile.close()
