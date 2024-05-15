import os
import numpy as np

if __name__ == '__main__':
    subjects = np.arange(1, 37)
    cond_names = ['tibial']
    for data_type in ['spinal', 'subcortical', 'cortical']:
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'

            for cond_name in cond_names:
                if data_type == 'spinal':
                    save_path = f"/data/pt_02718/tmp_data/cca_rs/{subject_id}/"
                elif data_type == 'cortical':
                    save_path = f"/data/pt_02718/tmp_data/cca_eeg_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    save_path = f"/data/pt_02718/tmp_data/cca_eeg_thalamic_rs/{subject_id}/"

                fname = f'{data_type}_stacked_task_{cond_name}.pkl'
                fname_new = f'{data_type}_stacked_task1_{cond_name}.pkl'
                os.rename(os.path.join(save_path, fname), os.path.join(save_path, fname_new))

                fname = f'{data_type}_stacked_rs_{cond_name}.pkl'
                fname_new = f'{data_type}_stacked_task_{cond_name}.pkl'
                os.rename(os.path.join(save_path, fname), os.path.join(save_path, fname_new))

                fname = f'{data_type}_stacked_task1_{cond_name}.pkl'
                fname_new = f'{data_type}_stacked_rs_{cond_name}.pkl'
                os.rename(os.path.join(save_path, fname), os.path.join(save_path, fname_new))

                print(f'Done {data_type}, {subject_id}, {cond_name}')