############################################################################################
# Take the 1000 CCA attempts in the task evoked and resting state (with fake triggers) data
# Crop eah evoked array of the 1000 to the time window used for CCA training
# Compute the correlation matrix, saving the image (heatmap) and the actual correlation
#############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo

if __name__ == '__main__':
    srmr_nr = 2
    sfreq = 1000

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'
        figure_folder = 'Images'
    else:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'
        figure_folder = 'Images_2'

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    for data_type in ['spinal', 'subcortical', 'cortical']:
        for condition in conditions:
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name

            for subject in subjects:
                # Set variables
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Select the right files
                if data_type == 'spinal':
                    input_path = f"/data/pt_02718/{folder}/cca_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    input_path = f"/data/pt_02718/{folder}/cca_eeg_thalamic_rs/{subject_id}/"
                elif data_type == 'cortical':
                    input_path = f"/data/pt_02718/{folder}/cca_eeg_rs/{subject_id}/"
                figure_path = f"/data/p_02718/{figure_folder}/TaskvsRest/{data_type}/"
                os.makedirs(figure_path, exist_ok=True)

                fname_trig = f"{data_type}_stacked_task_{cond_name}.pkl"
                fname_rest = f"{data_type}_stacked_rs_{cond_name}.pkl"

                # All evoked objects run from time -0.1s up to 0.299s
                # Have shape n_trials, n_times
                with open(f'{input_path}{fname_trig}', 'rb') as f:
                    stacked_trig = pickle.load(f)

                with open(f'{input_path}{fname_rest}', 'rb') as f:
                    stacked_rest = pickle.load(f)

                times = np.linspace(-0.1, 0.299, len(stacked_trig[0]))
                fig, ax = plt.subplots(1, 2, figsize=(21, 7))
                ax[0].plot(times, np.array(stacked_trig).T)
                ax[0].set_title('Task Evoked')
                ax[1].plot(times, np.array(stacked_rest).T)
                ax[1].set_title('Resting State')
                plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                plt.savefig(figure_path + f'{subject_id}_{cond_name}')
                plt.close()

                fig, ax = plt.subplots(1, 2, figsize=(21, 7))
                ax[0].plot(times, np.array(stacked_trig).T)
                ax[0].set_title('Task Evoked')
                ax[0].set_xlim([0, 0.065])
                ax[1].plot(times, np.array(stacked_rest).T)
                ax[1].set_title('Resting State')
                ax[1].set_xlim([0, 0.065])
                plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                plt.savefig(figure_path + f'{subject_id}_{cond_name}_shorter')
                plt.close()
                # plt.show()
                # exit()
