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
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.get_conditioninfo import get_conditioninfo

if __name__ == '__main__':
    freq_band = 'sigma'
    srmr_nr = 2
    iterations = 1000

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
                    input_path = f"/data/pt_02718/{folder}/cca_rs/{subject_id}/"
                elif data_type == 'cortical':
                    input_path = f"/data/pt_02718/{folder}/cca_eeg_rs/{subject_id}/"
                elif data_type == 'subcortical':
                    input_path = f"/data/pt_02718/{folder}/cca_eeg_thalamic_rs/{subject_id}/"
                figure_path = f"/data/p_02718/{figure_folder}/CCA_RS_Task_CCAWindow/{data_type}/"
                os.makedirs(figure_path, exist_ok=True)

                fname_trig = f"{data_type}_stacked_task_{cond_name}.pkl"
                fname_rest = f"{data_type}_stacked_rs_{cond_name}.pkl"

                # All evoked objects run from time -0.1s up to 0.299s
                # Have shape n_trials, n_times
                # Need to crop to the time windows above
                # Our times are in steps of 0.0002 s (5000Hz sample rate)
                # These are lists of numpy.ndarray
                with open(f'{input_path}{fname_trig}', 'rb') as f:
                    stacked_trig = pickle.load(f)

                with open(f'{input_path}{fname_rest}', 'rb') as f:
                    stacked_rest = pickle.load(f)

                # Now checking 10ms before and 80ms after window time
                indices = [int((5000*0.1) + (window_times[0])*5000), int((5000*0.1) + (window_times[1])*5000)]
                stacked_trig_shorter = [stacked_trig[i][indices[0]:indices[1]] for i in range(len(stacked_trig))]
                stacked_rest_shorter = [stacked_rest[i][indices[0]:indices[1]] for i in range(len(stacked_rest))]

                # Get correlation in shorter window
                R1 = np.corrcoef(stacked_trig_shorter)
                fig, ax = plt.subplots(1, 2, figsize=(21, 7))
                c = ax[0].pcolor(abs(R1), vmin=0, vmax=1)
                ax[0].set_title('Task Evoked')
                fig.colorbar(c, ax=ax[0])
                R2 = np.corrcoef(stacked_rest_shorter)
                c = ax[1].pcolor(abs(R2), vmin=0, vmax=1)
                ax[1].set_title('Resting State')
                fig.colorbar(c, ax=ax[1])
                plt.suptitle(f'{subject_id}, {data_type}, {cond_name}')
                plt.savefig(figure_path + f'{subject_id}_{cond_name}')
                plt.close()

                # Save correlation matrices
                afile = open(input_path + f'{data_type}_corr_task_ccawin_{cond_name}.pkl', 'wb')
                pickle.dump(abs(R1), afile)
                afile.close()

                afile = open(input_path + f'{data_type}_corr_rs_ccawin_{cond_name}.pkl', 'wb')
                pickle.dump(abs(R2), afile)
                afile.close()