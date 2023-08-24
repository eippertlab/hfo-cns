# Plot grand average envelope after application of CCA on EEG data
# Can use only the subjects with visible HFOs, or all subjects regardless


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    use_updated = True
    use_visible = True

    if use_updated is True and use_visible is not True:
        print('Error: If use_updated is True, use_visible must be True')
        exit()

    srmr_nr = 2

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        freq_bands = ['sigma']
    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        freq_bands = ['sigma']


    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    if use_updated:
        if srmr_nr == 1:
            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG_Updated.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Images/CCA_eeg/Envelope_Updated/'
            os.makedirs(figure_path, exist_ok=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)
        elif srmr_nr == 2:
            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Updated.xlsx')
            df = pd.read_excel(xls, 'CCA')
            df.set_index('Subject', inplace=True)

            figure_path = '/data/p_02718/Images_2/CCA_eeg/Envelope_Updated/'
            os.makedirs(figure_path, exist_ok=True)

            xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Updated.xlsx')
            df_vis = pd.read_excel(xls, 'CCA_Brain')
            df_vis.set_index('Subject', inplace=True)

    else:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_EEG.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA_eeg/Envelope/'
        os.makedirs(figure_path, exist_ok=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Brain')
        df_vis.set_index('Subject', inplace=True)

    for freq_band in freq_bands:
        for condition in conditions:
            evoked_list = []
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
                fname = f"{freq_band}_{cond_name}.fif"
                if srmr_nr == 1:
                    input_path = "/data/pt_02718/tmp_data/cca_eeg/" + subject_id + "/"
                elif srmr_nr == 2:
                    input_path = "/data/pt_02718/tmp_data_2/cca_eeg/" + subject_id + "/"

                epochs = mne.read_epochs(input_path + fname, preload=True)

                # Check if bursts are marked as visible
                visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]

                # Need to pick channel based on excel sheet
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                if not np.isnan(channel_no):
                    channel = f'Cor{int(channel_no)}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    envelope = evoked.apply_hilbert(envelope=True)
                    # data = envelope.get_data()
                    # evoked_list.append(data)

                    if use_visible is True:
                        visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                        if visible == 'T':
                            data = envelope.get_data()
                            evoked_list.append(data)
                    else:
                        data = envelope.get_data()
                        evoked_list.append(data)

            # Get grand average across chosen epochs, and spatial patterns
            grand_average = np.mean(evoked_list, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(epochs.times, grand_average[0, :])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
            if cond_name == 'median':
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])

            if use_visible:
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}_visible.png')
                plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}_visible.pdf',
                            bbox_inches='tight', format="pdf")
            else:
                plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.png')
                plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                            bbox_inches='tight', format="pdf")
            plt.close(fig)
