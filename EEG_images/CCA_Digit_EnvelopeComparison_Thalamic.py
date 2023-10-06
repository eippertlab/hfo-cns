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
    srmr_nr = 2  # Always 2 for digit information

    subjects = np.arange(1, 25)
    conditions = [2]  # med_digits
    freq_band = 'sigma'

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_EEG_Thalamic_Updated_Digits.xlsx')
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    figure_path = '/data/p_02718/Images_2/CCA_eeg_thalamic_digits/Envelope_Comparison/'
    os.makedirs(figure_path, exist_ok=True)

    xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Thalamic_Updated_Digits.xlsx')
    df_vis = pd.read_excel(xls, 'CCA_Brain')
    df_vis.set_index('Subject', inplace=True)

    for condition in conditions:
        # Set variables
        cond_info = get_conditioninfo(condition, srmr_nr)
        cond_name = cond_info.cond_name
        trigger_names = cond_info.trigger_name
        evoked_list_1 = []
        evoked_list_2 = []
        evoked_list_12 = []
        for subject in subjects:
            subject_id = f'sub-{str(subject).zfill(3)}'

            ##########################################################
            # Time  Course Information
            ##########################################################
            # Select the right files
            fname = f"{freq_band}_{cond_name}.fif"
            input_path = "/data/pt_02718/tmp_data_2/cca_eeg_thalamic/" + subject_id + "/"
            epochs_all = mne.read_epochs(input_path + fname, preload=True)

            for trigger_name, evoked_list in zip(trigger_names, [evoked_list_1, evoked_list_2, evoked_list_12]):
                epochs = epochs_all[trigger_name]
                # Check if bursts are marked as visible
                visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]

                # Need to pick channel based on excel sheet
                channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                if visible == 'T':
                    channel = f'Cor{int(channel_no)}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    envelope = evoked.apply_hilbert(envelope=True)
                    data = envelope.get_data()
                    evoked_list.append(data)

        # Get grand average across chosen epochs, and spatial patterns
        grand_average_1 = np.mean(evoked_list_1, axis=0)
        grand_average_2 = np.mean(evoked_list_2, axis=0)
        grand_average_12 = np.mean(evoked_list_12, axis=0)
        grand_average_1plus2 = grand_average_1 + grand_average_2

        # Plot Time Course
        fig, ax = plt.subplots()
        ax.plot(epochs.times, grand_average_1[0, :], label='med1')
        ax.plot(epochs.times, grand_average_2[0, :], label='med2')
        ax.plot(epochs.times, grand_average_12[0, :], label='med12')
        ax.plot(epochs.times, grand_average_1plus2[0, :], label='med1+med2')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
        if cond_name == 'median':
            ax.set_xlim([0.0, 0.05])
        else:
            ax.set_xlim([0.0, 0.07])

        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.png')
        plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                    bbox_inches='tight', format="pdf")
        plt.close(fig)