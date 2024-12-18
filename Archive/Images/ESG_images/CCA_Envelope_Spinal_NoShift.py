# Plot envelope of grand average CCA components of the ESG data
# Can choose to use only subjects marked for visible bursting, or all subjects regardless


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    use_updated = True  # If true use components selected with criteria in mind
    use_only_good = True  # If true use only the subjects marked as visible
    # use_only_good must be true if use_updated is true

    if use_updated is True and use_only_good is not True:
        print('Error: use_only_good must be true if use_updated is true')
        exit()

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

    if use_updated:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Updated.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Updated.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA/Envelope_Updated/'
        os.makedirs(figure_path, exist_ok=True)
    else:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA/Envelope/'
        os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

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
                if use_only_good:
                    # Only perform if bursts marked as visible
                    visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                    if visible == 'T':
                        # Select the right files
                        fname = f"{freq_band}_{cond_name}.fif"
                        input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"

                        epochs = mne.read_epochs(input_path + fname, preload=True)

                        # Need to pick channel based on excel sheet
                        channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                        channel = f'Cor{int(channel_no)}'
                        inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                        epochs = epochs.pick_channels([channel])
                        if inv == 'T':
                            epochs.apply_function(invert, picks=channel)
                        evoked = epochs.copy().average()
                        evoked.crop(tmin=-0.06, tmax=0.07)
                        envelope = evoked.apply_hilbert(envelope=True)
                        data = envelope.get_data()
                        evoked_list.append(data)

                else:
                    # Select the right files
                    fname = f"{freq_band}_{cond_name}.fif"
                    input_path = "/data/pt_02718/tmp_data/cca/" + subject_id + "/"

                    epochs = mne.read_epochs(input_path + fname, preload=True)

                    # Need to pick channel based on excel sheet
                    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    channel = f'Cor{channel_no}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    evoked.crop(tmin=-0.06, tmax=0.07)
                    envelope = evoked.apply_hilbert(envelope=True)
                    data = envelope.get_data()
                    evoked_list.append(data)

            # Get grand average across chosen epochs, and spatial patterns
            grand_average = np.mean(evoked_list, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(evoked.times, grand_average[0, :])
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
            ax.set_ylabel('Amplitude')
            if cond_name == 'median':
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])

            plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}')
            plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                        bbox_inches='tight', format="pdf")
            plt.close(fig)

