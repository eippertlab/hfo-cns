# Plot single subject spatial patterns after application of CCA for HFOs
# Plot single subject spatial patterns for low frequency raw potentials


import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
import matplotlib.pyplot as plt
from Common_Functions.IsopotentialFunctions_CbarLabel import mrmr_esg_isopotentialplot
import matplotlib as mpl
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    use_visible = True  # Use only subjects with visible bursting

    subjects = np.arange(1, 37)
    conditions = [2, 3]
    freq_band = 'sigma'
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    eeg_chans, esg_chans, bipolar_chans = get_channels(1, False, False, srmr_nr)
    montage_path = '/data/pt_02718/'
    montage_name = 'electrode_montage_eeg_10_5.elp'
    montage = mne.channels.read_custom_montage(montage_path + montage_name)

    figure_path = '/data/p_02718/Images/EEG/SpatialTopographies_Cortical&Thalamic/'
    os.makedirs(figure_path, exist_ok=True)

    for timing in ['Cortical', 'Thalamic']:
        for condition in conditions:
            for subject in subjects:
                # Set variables
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                trigger_name = cond_info.trigger_name
                subject_id = f'sub-{str(subject).zfill(3)}'

                if trigger_name == 'Median - Stimulation':
                    if timing == 'Cortical':
                        time_points = [0.018, 0.019, 0.02, 0.021, 0.022]
                    else:
                        time_points = [0.012, 0.013, 0.014, 0.015, 0.016]
                else:
                    if timing == 'Cortical':
                        time_points = [0.037, 0.038, 0.039, 0.040, 0.041]
                    else:
                        time_points = [0.028, 0.029, 0.030, 0.031, 0.032]

                # HFO before CCA
                input_path = "/data/pt_02718/tmp_data/freq_banded_eeg/" + subject_id + "/"
                fname = f"{freq_band}_{cond_name}.fif"
                raw = mne.io.read_raw_fif(input_path + fname, preload=True)
                # Set montage
                raw.set_montage(montage, on_missing="ignore")
                idx_by_type = mne.channel_indices_by_type(raw.info, picks=eeg_chans)
                res = mne.pick_info(raw.info, sel=idx_by_type['eeg'], copy=True, verbose=None)
                # raw.plot_sensors(ch_type='eeg', show_names=True)
                evoked_high = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked_high = evoked_high.pick_channels(eeg_chans, ordered=True)

                # Low Freq SEP
                input_path_low = "/data/pt_02068/analysis/final/tmp_data/" + subject_id + "/eeg/prepro/"
                fname_low = f"cnt_clean_{cond_name}.set"
                raw = mne.io.read_raw_eeglab(input_path_low + fname_low, preload=True)
                raw.set_montage(montage, on_missing="ignore")
                evoked_low = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)
                evoked_low = evoked_low.pick_channels(eeg_chans, ordered=True)

                # Plot low and high frequency spatial topographies at the specified time points
                ###############################################################################################
                # High Freq SEP
                ###############################################################################################
                fig_high, ax_high = plt.subplots(1, len(time_points))
                evoked_high.plot_topomap(times=time_points, average=None, ch_type=None, scalings=None, proj=False,
                                      sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                      outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                      border='mean',
                                      res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None,
                                      cnorm=None,
                                      colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_high, time_unit='s',
                                      time_format=None,
                                      nrows=1, ncols='auto', show=False)
                fig_high.suptitle(f'{subject_id} Spatial Pattern HFO, {timing}, {cond_name}')
                divider = make_axes_locatable(ax_high[4])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = fig_high.colorbar(ax_high[4].images[-1], cax=cax, shrink=0.6, orientation='vertical')
                cb.set_label('Amplitude (\u03BCV)', rotation=90)

                ###############################################################################################
                # Low Freq SEP
                ###############################################################################################
                fig_low, ax_low = plt.subplots(1, len(time_points))
                evoked_low.plot_topomap(times=time_points, average=None, ch_type=None, scalings=None, proj=False,
                                      sensors=True, show_names=False, mask=None, mask_params=None, contours=6,
                                      outlines='head', sphere=None, image_interp='cubic', extrapolate='auto',
                                      border='mean',
                                      res=64, size=1, cmap='jet', vlim=(None, None), vmin=None, vmax=None,
                                      cnorm=None,
                                      colorbar=False, cbar_fmt='%3.1f', units=None, axes=ax_low, time_unit='s',
                                      time_format=None,
                                      nrows=1, ncols='auto', show=False)
                fig_low.suptitle(f'{subject_id} Spatial Pattern Low SEP, {timing}, {cond_name}')
                divider = make_axes_locatable(ax_low[4])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                cb = fig_low.colorbar(ax_low[4].images[-1], cax=cax, shrink=0.6, orientation='vertical')
                cb.set_label('Amplitude (\u03BCV)', rotation=90)

                fig_high.savefig(figure_path + f'{subject_id}_HFO_Spatial_{cond_name}_{timing}')
                # fig_high.savefig(figure_path + ff'{subject_id}_HFO_Spatial_{cond_name}_{timing}.pdf',
                #             bbox_inches='tight', format="pdf")

                fig_low.savefig(figure_path + f'{subject_id}_SEP_Spatial_{cond_name}_{timing}')
                # fig_low.savefig(figure_path + f'{subject_id}_SEP_Spatial_{cond_name}_{timing}.pdf',
                #                 bbox_inches='tight', format="pdf")
                plt.close()