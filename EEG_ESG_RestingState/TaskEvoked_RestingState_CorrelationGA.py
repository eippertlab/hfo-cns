# Want to take 50% of task evoked and resting state trials (resting state trials are dummy)
# Want to compute CCA for both
# Want to store time course of component 1
# Repeat 1000 times
# Get correlation between time courses - should be higher for task evoked versus resting state

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from Common_Functions.get_conditioninfo import get_conditioninfo

if __name__ == '__main__':
    freq_band = 'sigma'
    srmr_nr = 1
    iterations = 1000

    if srmr_nr == 1:
        subjects = np.arange(1, 11)  # Haven't generated all RS data yet
        conditions = [2, 3]
    else:
        raise RuntimeError('Only implemented for srmr_nr 1')

    for data_type in ['spinal', 'subcortical', 'cortical']:
        for condition in conditions:
            resting_state = np.empty((len(subjects), 1000, 1000))  # Iterations by timepoints
            task_evoked = np.empty((len(subjects), 1000, 1000))

            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Select the right files
                if data_type == 'spinal':
                    input_path = f"/data/pt_02718/tmp_data/cca_rs/{subject_id}/"
                elif data_type == 'cortical':
                    input_path = f"/data/pt_02718/tmp_data/cca_eeg_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    input_path = f"/data/pt_02718/tmp_data/cca_eeg_thalamic_rs/{subject_id}/"
                figure_path = f"/data/p_02718/Images/CCA_RS_Task/{data_type}/GA/"
                os.makedirs(figure_path, exist_ok=True)

                # Load subjects correlation matrices and append
                with open(f'{input_path}{data_type}_corr_task_{cond_name}.pkl', 'rb') as f:
                    R1 = pickle.load(f)
                task_evoked[subject-1, :, :] = R1

                with open(f'{input_path}{data_type}_corr_rs_{cond_name}.pkl', 'rb') as f:
                    R2 = pickle.load(f)
                resting_state[subject-1, :, :] = R2

            fig, ax = plt.subplots(1, 2, figsize=(21, 7))
            c = ax[0].pcolor(task_evoked.mean(axis=0), vmin=-1, vmax=1)
            ax[0].set_title('Task Evoked')
            fig.colorbar(c, ax=ax[0])


            c = ax[1].pcolor(resting_state.mean(axis=0), vmin=-1, vmax=1)
            ax[1].set_title('Resting State')
            fig.colorbar(c, ax=ax[1])
            plt.suptitle(f'GA n={len(subjects)}, {data_type}, {cond_name}')
            plt.savefig(figure_path+f'{cond_name}_n={len(subjects)}')
            plt.close()