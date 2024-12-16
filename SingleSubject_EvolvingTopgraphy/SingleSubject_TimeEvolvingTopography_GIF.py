# Testing only in subject 6, median as they have high SNR HFOs

import os
import mne
import numpy as np
from meet import spatfilt
from scipy.io import loadmat
import imageio
from PIL import Image
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.get_channels import get_channels
from Common_Functions.get_esg_channels import get_esg_channels
import matplotlib.pyplot as plt
from Common_Functions.IsopotentialFunctions_CbarLabel import mrmr_esg_isopotentialplot
from Common_Functions.evoked_from_raw import evoked_from_raw
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_gif(image_paths, output_gif_path, duration=500):
 images = [Image.open(image_path) for image_path in image_paths]
# Save as GIF
 images[0].save(
 output_gif_path,
 save_all=True,
 append_images=images[1:],
 duration=duration,
 loop=0 # 0 means infinite loop
 )

if __name__ == '__main__':
    subject = 36
    condition = 2
    srmr_nr = 1

    cfg_path = "/data/pt_02718/cfg.xlsx"  # Contains important info about experiment
    df = pd.read_excel(cfg_path)
    iv_baseline = [df.loc[df['var_name'] == 'baseline_start', 'var_value'].iloc[0],
                   df.loc[df['var_name'] == 'baseline_end', 'var_value'].iloc[0]]
    iv_epoch = [df.loc[df['var_name'] == 'epo_cca_start', 'var_value'].iloc[0],
                df.loc[df['var_name'] == 'epo_cca_end', 'var_value'].iloc[0]]
    iv_crop = [df.loc[df['var_name'] == 'epoch_start', 'var_value'].iloc[0],
               df.loc[df['var_name'] == 'epoch_end', 'var_value'].iloc[0]]

    figure_path = '/data/p_02718/Polished/SingleSubject_EvolvingSpatialPatterns/'
    os.makedirs(figure_path, exist_ok=True)
    brainstem_chans, cervical_chans, lumbar_chans, ref_chan = get_esg_channels()
    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

    # Set variables
    cond_info = get_conditioninfo(condition, srmr_nr)
    cond_name = cond_info.cond_name
    trigger_name = cond_info.trigger_name
    subject_id = f'sub-{str(subject).zfill(3)}'

    # HFO before CCA
    fname = f"sigma_{cond_name}.fif"
    input_path = f"/data/pt_02718/tmp_data/freq_banded_esg/{subject_id}/"
    raw = mne.io.read_raw_fif(input_path + fname, preload=True)
    evoked = evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, False)

    chan_labels = esg_chans
    colorbar_axes = [-0.1, 0.1]
    subjects_4grid = np.arange(1, 37)
    # then the function takes the average over the channel positions of all those subjects
    filenames = []
    time_points = [8/1000, 8.5/1000, 9/1000, 9.5/1000, 10/1000, 10.5/1000, 11/1000, 11.5/1000,
                   12/1000, 12.5/1000, 13/1000, 13.5/1000, 14/1000, 14.5/1000, 15/1000, 15.5/1000]
    for iplot in np.arange(0, len(time_points)):
        fig, axis = plt.subplots(1, 1)
        time_point = time_points[iplot]
        # Extracts value at time point only
        if iplot == 0:
            colorbar = True
        else:
            colorbar = False
        chanvalues = evoked.copy().pick(esg_chans).get_data(tmin=time_point, tmax=time_point+2/10000)
        mrmr_esg_isopotentialplot(subjects_4grid, chanvalues, colorbar_axes, chan_labels, colorbar,
                                  time_point, axis, colorbar_label='Amplitude (\u03BCV)',
                                  srmr_nr=srmr_nr)
        axis.set_yticklabels([])
        axis.set_ylabel(None)
        axis.set_xticklabels([])
        axis.set_xlabel(None)
        axis.set_title(f"{time_point}s")
        filename = f'{time_point}s.png'
        filenames.append(filename)
        # last frame of each viz stays longer - stops the gif being too fast
        # for i in range(250):
        #
        filenames.append(filename)
        plt.savefig(filename)

    # Build Gif
    save_name = f'{figure_path}{subject_id}_{cond_name}.gif'
    # # imageio.mimsave(save_name, filenames, duration=0.5)
    # with imageio.get_writer(save_name, mode='I') as writer:
    #     for filename in filenames:
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    create_gif(filenames, save_name)

    # Remove files
    for filename in set(filenames):
        os.remove(filename)

    plt.close()

