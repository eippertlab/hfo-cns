# Plot envelope of grand average CCA components of the ESG data
# Can choose to use only subjects marked for visible bursting, or all subjects regardless
# Shifting the envelope based on my updated estimates of the spinal timing


import os
import mne
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
from Common_Functions.invert import invert
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from Common_Functions.GetTimeToAlign_Old import get_time_to_align
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    srmr_nr = 1

    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2]  # 4: Don't do for tibial - no surviving subjects
        freq_bands = ['sigma']
    elif srmr_nr == 2:
        print('Not yet implemented')
        exit()
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        freq_bands = ['sigma']

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    if srmr_nr == 1:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components_Shuffle_Updated.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility_Shuffle_Updated.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data/LowFreq_HighFreq_Relation.xlsx')
        df_timing = pd.read_excel(xls_timing, 'Spinal')
        df_timing.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images/CCA_shuffle/Envelope_Updated_Shifted/'
        os.makedirs(figure_path, exist_ok=True)

    elif srmr_nr == 2:
        xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Components_Shuffle_Updated.xlsx')
        df = pd.read_excel(xls, 'CCA')
        df.set_index('Subject', inplace=True)

        xls = pd.ExcelFile('/data/pt_02718/tmp_data_2/Visibility_Shuffle_Updated.xlsx')
        df_vis = pd.read_excel(xls, 'CCA_Spinal')
        df_vis.set_index('Subject', inplace=True)

        xls_timing = pd.ExcelFile('/data/pt_02718/tmp_data_2/LowFreq_HighFreq_Relation.xlsx')
        df_timing = pd.read_excel(xls_timing, 'Spinal')
        df_timing.set_index('Subject', inplace=True)

        figure_path = '/data/p_02718/Images_2/CCA_shuffle/Envelope_Updated_Shifted/'
        os.makedirs(figure_path, exist_ok=True)

    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    median_names = ['median', 'med_mixed']
    tibial_names = ['tibial', 'tib_mixed']

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
                # Align subject with average latency across all subjects
                if srmr_nr == 1:
                    condition_names = ['median', 'tibial']
                elif srmr_nr == 2:
                    condition_names = ['med_mixed', 'tib_mixed']
                # median_lat, tibial_lat = get_time_to_align('esg', srmr_nr, condition_names, subjects)
                median_lat = 0.013
                tibial_lat = 0.022
                if cond_name in median_names:
                    sep_latency = round(df_timing.loc[subject, f"N13"], 3)
                    expected = median_lat
                elif cond_name in tibial_names:
                    sep_latency = round(df_timing.loc[subject, f"N22"], 3)
                    expected = tibial_lat
                shift = expected - sep_latency

                # Only perform if bursts marked as visible
                visible = df_vis.loc[subject, f"{freq_band.capitalize()}_{cond_name.capitalize()}_Visible"]
                print(visible)
                if visible == 'T':
                    # Select the right files
                    fname = f"{freq_band}_{cond_name}.fif"
                    if srmr_nr == 1:
                        input_path = "/data/pt_02718/tmp_data/cca_shuffle/" + subject_id + "/"
                    elif srmr_nr == 2:
                        input_path = "/data/pt_02718/tmp_data_2/cca_shuffle/" + subject_id + "/"

                    epochs = mne.read_epochs(input_path + fname, preload=True)

                    # Need to pick channel based on excel sheet
                    channel_no = df.loc[subject, f"{freq_band}_{cond_name}_comp"]
                    channel = f'Cor{int(channel_no)}'
                    inv = df.loc[subject, f"{freq_band}_{cond_name}_flip"]
                    epochs = epochs.pick_channels([channel])
                    if inv == 'T':
                        epochs.apply_function(invert, picks=channel)
                    evoked = epochs.copy().average()
                    evoked.shift_time(shift, relative=True)
                    evoked.crop(tmin=-0.06, tmax=0.07)
                    envelope = evoked.apply_hilbert(envelope=True)
                    data = envelope.get_data()
                    evoked_list.append(data)

            # Get grand average across chosen epochs
            grand_average = np.mean(evoked_list, axis=0)

            # Plot Time Course
            fig, ax = plt.subplots()
            ax.plot(evoked.times, grand_average[0, :])
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Grand Average Envelope, n={len(evoked_list)}')
            ax.set_ylabel('Amplitude')
            if cond_name in median_names:
                ax.set_xlim([0.0, 0.05])
            else:
                ax.set_xlim([0.0, 0.07])

            plt.savefig(figure_path+f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}')
            plt.savefig(figure_path + f'GA_Envelope_{freq_band}_{cond_name}_n={len(evoked_list)}.pdf',
                        bbox_inches='tight', format="pdf")
            plt.close(fig)
