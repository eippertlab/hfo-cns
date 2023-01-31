import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import pickle

if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_band = 'sigma'
    srmr_nr = 1

    for condition in conditions:
        for subject in subjects:
            # Set variables
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            subject_id = f'sub-{str(subject).zfill(3)}'

            potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
            cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
            df = pd.read_excel(cfg_path)
            iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                           df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
            iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                        df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

            # Select the right files based on the data_string
            input_path = "/data/pt_02718/tmp_data/freq_banded_esg/" + subject_id + "/"
            input_path_good = "/data/pt_02718/tmp_data/good_trials_spinal/" + subject_id + "/"
            fname = f"{freq_band}_{cond_name}.fif"
            save_path = "/data/p_02718/Images/ESG/WrongChannel/"
            os.makedirs(save_path, exist_ok=True)

            esg_chans = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                         'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                         'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                         'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                         'S23', 'TH6']

            brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

            raw = mne.io.read_raw_fif(input_path + fname, preload=True)

            # now create epochs based on the trigger names
            events, event_ids = mne.events_from_annotations(raw)
            event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
            epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1]-1/1000,
                                baseline=tuple(iv_baseline), preload=True)

            # cca window size - Birgit created individual potential latencies for each subject
            fname_pot = 'potential_latency.mat'
            matdata = loadmat(potential_path + fname_pot)

            # Get good trials
            # Read in saved A_st
            with open(f'{input_path_good}good_{freq_band}_{cond_name}_strict.pkl', 'rb') as f:
                vals = pickle.load(f)
                drop_bad = [idx for idx, element in enumerate(vals) if element == False]
                epochs_good = epochs.copy().drop(drop_bad)

            if cond_name == 'median':
                sep_latency = matdata['med_potlatency']
                channel_correct = 'SC6'
                channel_incorrect = 'L1'
            elif cond_name == 'tibial':
                sep_latency = matdata['tib_potlatency']
                channel_correct = 'L1'
                channel_incorrect = 'SC6'
            else:
                print('Invalid condition name attempted for use')
                exit()

            evoked_good_cor = epochs_good.average().pick_channels([channel_correct])
            evoked_cor = epochs.average().pick_channels([channel_correct])

            evoked_good_incor = epochs_good.average().pick_channels([channel_incorrect])
            evoked_incor = epochs.average().pick_channels([channel_incorrect])

            for string in ['All', 'GoodOnly']:
                if string == 'All':
                    plt.figure()
                    plt.plot(epochs.times, evoked_cor.get_data().reshape(-1), label='Correct')
                    plt.plot(epochs.times, evoked_incor.get_data().reshape(-1), label='Incorrect')
                    line_label = f"{sep_latency[0][0] / 1000}s"
                    plt.axvline(x=sep_latency[0][0] / 1000, color='r', linewidth='0.6', label=line_label)
                    plt.xlim([-0.025, 0.065])
                    plt.title(f'{subject_id}_{cond_name}_{freq_band}_AllTrials')
                    plt.legend()
                    plt.savefig(save_path+f'{subject_id}_{cond_name}_{freq_band}_AllTrials.png')
                    plt.close()
                else:
                    plt.figure()
                    plt.plot(epochs.times, evoked_good_cor.get_data().reshape(-1), label='Correct')
                    plt.plot(epochs.times, evoked_good_incor.get_data().reshape(-1), label='Incorrect')
                    line_label = f"{sep_latency[0][0] / 1000}s"
                    plt.axvline(x=sep_latency[0][0] / 1000, color='r', linewidth='0.6', label=line_label)
                    plt.xlim([-0.025, 0.065])
                    plt.title(f'{subject_id}_{cond_name}_{freq_band}_GoodTrials')
                    plt.legend()
                    plt.savefig(save_path + f'{subject_id}_{cond_name}_{freq_band}_GoodTrials.png')
                    plt.close()
            # plt.show()
