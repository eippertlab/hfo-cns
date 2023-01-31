# Plot envelope of grand average CCA components


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.IsopotentialFunctions import mrmr_esg_isopotentialplot
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


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

    figure_path = '/data/p_02718/Images/CCA_SpatialSpecificity/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    use_average = False

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list_correct = []
            evoked_list_incorrect = []

            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                # Select the right files
                fname = f"{freq_band}_{cond_name}.fif"
                input_path_correct = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"
                input_path_incorrect = "/data/pt_02718/tmp_data/cca_opposite/" + subject_id + "/"

                ##########################################################
                # Time  Course Information
                ##########################################################
                epochs_correct = mne.read_epochs(input_path_correct + fname, preload=True)
                epochs_incorrect = mne.read_epochs(input_path_incorrect + fname, preload=True)

                # Need to pick channel based on excel sheet
                if use_average:
                    for epochs, evoked_list in \
                            zip([epochs_correct, epochs_incorrect], [evoked_list_correct, evoked_list_incorrect]):
                        channel = ['Cor1', 'Cor2']
                        epochs = epochs.pick_channels(channel)
                        evoked = epochs.copy().average()
                        envelope = evoked.apply_hilbert(envelope=True)
                        data = envelope.get_data().mean(axis=0).reshape(1, len(epochs.times))
                        evoked_list.append(data)
                else:
                    for epochs, evoked_list in \
                            zip([epochs_correct, epochs_incorrect], [evoked_list_correct, evoked_list_incorrect]):
                        channel = 'Cor1'
                        epochs = epochs.pick_channels([channel])
                        evoked = epochs.copy().average()
                        envelope = evoked.apply_hilbert(envelope=True)
                        data = envelope.get_data()
                        evoked_list.append(data)

            # Get grand average across chosen epochs, and spatial patterns
            grand_average_correct = np.mean(evoked_list_correct, axis=0)
            grand_average_incorrect = np.mean(evoked_list_incorrect, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(epochs.times, grand_average_correct[0, :], label='Correct Patch')
            ax.plot(epochs.times, grand_average_incorrect[0, :], label='Incorrect Patch')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
            ax.set_ylabel('Amplitude')
            if cond_name == 'median':
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])
            plt.legend()

            if use_average:
                plt.savefig(figure_path+f'Average_GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}')
                # plt.savefig(figure_path + f'Average_GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                #             bbox_inches='tight', format="pdf")
                plt.close(fig)
            else:
                plt.savefig(figure_path + f'Cor1_GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}')
                # plt.savefig(figure_path + f'Cor1_GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                #             bbox_inches='tight', format="pdf")
                plt.close(fig)
