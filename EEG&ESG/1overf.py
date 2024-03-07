# Want to explore the aperiodic content between 400 and 800Hz to see if this relates to excluded subjects

import mne
from Common_Functions.get_channels import get_channels
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.check_excel_exist_1overf import check_excel_exist_1overf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# FOOOF imports
from fooof import FOOOFGroup
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectra
from fooof.plts.annotate import plot_annotated_model

def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data

if __name__ == '__main__':
    # Want to compute PSD for 50Hz further than range of interest to avoid edge effects impacting slope fit
    fmin_comp = 350
    fmax_comp = 850
    fmin_slope = 400
    fmax_slope = 800

    for srmr_nr in [1, 2]:
        if srmr_nr == 1:
            subjects = np.arange(1, 37)
            conditions = [2, 3]
            add_on = ''

        elif srmr_nr == 2:
            subjects = np.arange(1, 25)
            conditions = [3, 5]
            add_on = '_2'

        excel_fname = f'/data/pt_02718/tmp_data{add_on}/1overf.xlsx'

        for data_type in ['Spinal', 'Cortical']:
            # Check all is well for writing results
            sheetname = data_type
            check_excel_exist_1overf(subjects, srmr_nr, excel_fname, sheetname)
            df_rel = pd.read_excel(excel_fname, sheetname)
            df_rel.set_index('Subject', inplace=True)

            for condition in conditions:
                cond_info = get_conditioninfo(condition, srmr_nr)
                cond_name = cond_info.cond_name
                for subject in subjects:
                    subject_id = f'sub-{str(subject).zfill(3)}'
                    eeg_chans, esg_chans, bipolar_chans = get_channels(subject, False, False, srmr_nr)

                    if data_type == 'Spinal':
                        input_path = f'/data/pt_02718/tmp_data{add_on}/freq_banded_esg/{subject_id}/'
                        if cond_name in ['median', 'med_mixed']:
                            ch = 'SC6'
                        else:
                            ch = 'L1'
                    elif data_type == 'Cortical':
                        input_path = f'/data/pt_02718/tmp_data{add_on}/freq_banded_eeg/{subject_id}/'
                        if cond_name in ['median', 'med_mixed']:
                            ch = 'CP4'
                        else:
                            ch = 'Cz'

                    fname = f'sigma_{cond_name}.fif'
                    raw = mne.io.read_raw_fif(input_path+fname, preload=True, verbose=False)
                    raw = raw.pick(ch)

                    # Calculate power spectra across the continuous data
                    psd = raw.compute_psd(method="welch", fmin=fmin_comp, fmax=fmax_comp,
                                          n_overlap=150, n_fft=300)
                    spectra, freqs = psd.get_data(return_freqs=True)

                    # Initialize a FOOOFGroup object, with desired settings
                    fg = FOOOFGroup(peak_width_limits=[1, 12], min_peak_height=0.0,
                                    peak_threshold=2., verbose=False, aperiodic_mode='fixed')  # All default values

                    # Define the frequency range to fit
                    freq_range = [fmin_slope, fmax_slope]

                    # Fit the power spectrum model across all channels
                    fg.fit(freqs, spectra, freq_range)

                    # Check the aperiodic parameters
                    # print(fg.get_fooof(0).aperiodic_params_)  # offset, exponent
                    df_rel.at[subject, f'{cond_name}_offset'] = fg.get_fooof(0).aperiodic_params_[0]
                    df_rel.at[subject, f'{cond_name}_exponent'] = fg.get_fooof(0).aperiodic_params_[1]

            # Write the dataframe to the excel file
            with pd.ExcelWriter(excel_fname, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
                df_rel.to_excel(writer, sheet_name=sheetname)

    ############################################################################################################
    # Test Plotting
    ############################################################################################################

    # # Check the overall results of the group fits
    # fg.plot()
    # fig, axes = plt.subplots(1, 1, figsize=(15, 6))
    # bands = Bands({'sigma': [400, 800]})
    # # Get the power values across channels for the current band
    # band_power = check_nans(get_band_peak_fg(fg, bands['sigma'])[:, 1])
    #
    # # Extracted and plot the power spectrum model with the most band power
    # fg.get_fooof(0).plot(ax=axes, add_legend=False)
    #
    # # Set some plot aesthetics & plot title
    # axes.yaxis.set_ticklabels([])
    # axes.set_title('biggest peak', {'fontsize': 16})
    #
    # # Plot annotated model of aperiodic parameters
    # plot_annotated_model(fg.get_fooof(0), annotate_peaks=False, annotate_aperiodic=True, plt_log=True)
    #
    # plt.show()