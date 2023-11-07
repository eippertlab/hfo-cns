# Plot envelope of grand average CCA components for ESG data
# Shift all subjects depending on the peak of their low-frequency potential


import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_esg_channels import get_esg_channels
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
    freq_bands = ['sigma', 'kappa']
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]

    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Components.xlsx')
    df = pd.read_excel(xls, 'CCA')
    df.set_index('Subject', inplace=True)

    xls = pd.ExcelFile('/data/pt_02718/BurstPeaks/Peaks_from_Envelope.xlsx')
    df_vis = pd.read_excel(xls, 'Sheet1')
    df_vis.set_index('Subjects', inplace=True)

    figure_path = '/data/p_02718/Images/CCA/EnvelopeShift/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()

    use_only_good = True

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
                    visible = df_vis.loc[subject, f"{freq_band}_{cond_name}_visible"]
                    double = df_vis.loc[subject, f"{freq_band}_{cond_name}_double"]
                    if visible == 'T' and double == 'F':
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

                        if cond_name == 'median':
                            sep_latency = df_vis.loc[subject, f"{freq_band}_{cond_name}"]
                            expected = 13 / 1000
                        elif cond_name == 'tibial':
                            sep_latency = df_vis.loc[subject, f"{freq_band}_{cond_name}"]
                            expected = 22 / 1000
                        shift = sep_latency - expected
                        evoked = epochs.copy().average()
                        evoked.shift_time(shift, relative=True)
                        evoked.crop(tmin=-0.06, tmax=0.20)
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
                    if cond_name == 'median':
                        sep_latency = df_vis.loc[subject, f"{freq_band}_{cond_name}"]
                        expected = 13 / 1000
                    elif cond_name == 'tibial':
                        sep_latency = df_vis.loc[subject, f"{freq_band}_{cond_name}"]
                        expected = 22 / 1000
                    shift = sep_latency - expected
                    evoked = epochs.copy().average()
                    evoked.shift_time(shift, relative=True)
                    evoked.crop(tmin=-0.06, tmax=0.20)
                    envelope = evoked.apply_hilbert(envelope=True)
                    data = envelope.get_data()
                    evoked_list.append(data)

            # Get grand average across chosen epochs, and spatial patterns
            grand_average = np.mean(evoked_list, axis=0)

            # Plot Time Course of Envelope
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

