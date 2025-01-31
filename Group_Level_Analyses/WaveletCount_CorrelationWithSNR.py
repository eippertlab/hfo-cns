# Want to see if the number of burst peaks counted is related to the SNR of the HFO burst

import mne
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    srmr_nr = 1
    sfreq = 5000
    freq_band = 'sigma'

    ##############################################################################################################
    # Set paths and variables
    ##############################################################################################################
    if srmr_nr == 1:
        subjects = np.arange(1, 37)
        conditions = [2, 3]
        folder = 'tmp_data'

    elif srmr_nr == 2:
        subjects = np.arange(1, 25)
        conditions = [3, 5]
        folder = 'tmp_data_2'

    # Wavelet Counts - obtained by adding peaks and troughs and dividing by 2
    excel_path = f"/data/pt_02718/{folder}/"

    # Read in SNR excel files
    excel_fname = f"{excel_path}LowFreq_HighFreq_Amp_SNR_CCA.xlsx"
    for sheetname in ['Cortical', 'Spinal']:
        if sheetname == 'Cortical':
            col_names_cort = ['Subject', 'N20', 'N20_amplitude', 'N20_high', 'N20_high_amplitude',
                             'N20_SNR', 'N20_high_SNR',
                             'P39', 'P39_amplitude', 'P39_high', 'P39_high_amplitude',
                             'P39_SNR', 'P39_high_SNR']
            df_snr_cort = pd.read_excel(excel_fname, sheetname)
            df_snr_cort.set_index('Subject', inplace=True)
        elif sheetname == 'Spinal':
            col_names_spin = ['Subject', 'N13', 'N13_amplitude', 'N13_high', 'N13_high_amplitude',
                             'N13_SNR', 'N13_high_SNR',
                             'N22', 'N22_amplitude', 'N22_high', 'N22_high_amplitude',
                             'N22_SNR', 'N22_high_SNR']
            df_snr_spin = pd.read_excel(excel_fname, sheetname)
            df_snr_spin.set_index('Subject', inplace=True)

    # Read in peaks_troughs excel file, compute wavelet count and compare to SNR
    excel_fname = f'{excel_path}Peaks_Troughs_EqualWindow.xlsx'
    for sheetname in ['10%', '20%', '25%', '33%', '50%']:
        df_pt = pd.read_excel(excel_fname, sheetname)
        df_pt.set_index('Subject', inplace=True)

        for pot_name in ['N13', 'N22', 'N20', 'P39']:
                if pot_name == 'N13':
                    data = 'spinal'
                    cond = 'median'
                    df_snr = df_snr_spin
                elif pot_name == 'N22':
                    data = 'spinal'
                    cond = 'tibial'
                    df_snr = df_snr_spin
                elif pot_name == 'N20':
                    data = 'cortical'
                    cond = 'median'
                    df_snr = df_snr_cort
                elif pot_name == 'P39':
                    data = 'cortical'
                    cond = 'tibial'
                    df_snr = df_snr_cort

                df_new = pd.DataFrame()
                df_new["snr_low"] = df_snr[f'{pot_name}_SNR']
                df_new["snr_high"] = df_snr[f'{pot_name}_high_SNR']
                peaks = df_pt[f"{data}_peaks_{cond}"]
                troughs = df_pt[f"{data}_troughs_{cond}"]
                df_new["wavelet_count"] = (peaks+troughs)/2
                fig, axes = plt.subplots(2, 1)
                axes = axes.flatten()
                sns.scatterplot(x="wavelet_count", y="snr_low", data=df_new, ax=axes[0])
                sns.scatterplot(x="wavelet_count", y="snr_high", data=df_new, ax=axes[1])
                plt.suptitle(f"{sheetname} threshold, {pot_name}")
                plt.tight_layout()
                plt.show()
                plt.close()
