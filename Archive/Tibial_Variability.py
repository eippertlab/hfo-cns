# Is N22 more variable than N13?
# Do subjects with more N22 variability have worse SNR - check from project 1 - check single subject images too
# Do subjects with bad HFOs have greater cross trial variability

from scipy.io import loadmat
import numpy as np
from Common_Functions.get_conditioninfo import get_conditioninfo
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sns


if __name__ == '__main__':
    subjects = np.arange(1, 37)
    conditions = [2, 3]
    srmr_nr = 1

    df = pd.DataFrame()
    df['Subjects'] = subjects
    tibial_timing = []
    median_timing = []

    for condition in conditions:
        evoked_list = []
        spatial_pattern = []
        for subject in subjects:
            # Set variables
            cond_info = get_conditioninfo(condition, srmr_nr)
            cond_name = cond_info.cond_name
            trigger_name = cond_info.trigger_name
            subject_id = f'sub-{str(subject).zfill(3)}'

            potential_path = f"/data/p_02068/SRMR1_experiment/analyzed_data/esg/{subject_id}/"
            fname_pot = 'potential_latency.mat'
            matdata = loadmat(potential_path + fname_pot)
            if cond_name == 'median':
                sep_latency = matdata['med_potlatency'][0][0]
                expected = 13
                median_timing.append(sep_latency-expected)

            elif cond_name == 'tibial':
                sep_latency = matdata['tib_potlatency'][0][0]
                expected = 22
                tibial_timing.append(sep_latency-expected)

    print(f"Median Mean difference to expected: {np.mean(np.abs(median_timing)):.2f}")
    print(f"Median Std difference to expected: {np.std(np.abs(median_timing)):.2f}")

    print(f"Median Mean difference to expected: {np.mean(np.abs(tibial_timing)):.2f}")
    print(f"Median Std difference to expected: {np.std(np.abs(tibial_timing)):.2f}")

    df['Median Difference'] = np.abs(median_timing)
    df['Tibial Difference'] = np.abs(tibial_timing)

    ######################################################################################
    # Get the SNR info from Cardiac Artefact Removal Project
    ######################################################################################
    keywords = ['snr_med', 'snr_tib']
    input_path = "/data/p_02569/SSP/"
    name = 'SSP'
    snr_list_med = {}
    snr_list_tib = {}
    fn = f"{input_path}snr.h5"

    # SSP is (36, n_proj)
    with h5py.File(fn, "r") as infile:
        if name == 'SSP':
            snr_med = infile[keywords[0]][()]
            snr_tib = infile[keywords[1]][()]
            for i, (data_med, data_tib) in enumerate(zip(snr_med.T, snr_tib.T)):
                snr_list_med[f'{name}_{i + 1}'] = data_med
                snr_list_tib[f'{name}_{i + 1}'] = data_tib

    df_med = pd.DataFrame(snr_list_med)
    df_med = df_med[['SSP_6']]
    df_tib = pd.DataFrame(snr_list_tib)
    df_tib = df_tib[['SSP_6']]

    df['Median SNR'] = pd.Series(df_med['SSP_6'])
    df['Tibial SNR'] = pd.Series(df_tib['SSP_6'])

    #############################################################################################
    # Load in which have visible HFOs
    ##############################################################################################
    xls = pd.ExcelFile('/data/pt_02718/tmp_data/Visibility.xlsx')
    df_vis = pd.read_excel(xls, 'CCA_Spinal')

    df['Sigma_Median_Visible'] = pd.Series(df_vis['Sigma_Median_Visible'])
    df['Kappa_Median_Visible'] = pd.Series(df_vis['Kappa_Median_Visible'])
    df['Sigma_Tibial_Visible'] = pd.Series(df_vis['Sigma_Tibial_Visible'])
    df['Kappa_Tibial_Visible'] = pd.Series(df_vis['Kappa_Tibial_Visible'])

    # Rearrange order to make easier
    df = df[['Subjects', 'Median Difference', 'Median SNR', 'Sigma_Median_Visible', 'Kappa_Median_Visible',
             'Tibial Difference', 'Tibial SNR', 'Sigma_Tibial_Visible', 'Kappa_Tibial_Visible']]

    ##############################################################################################
    # Bar plot
    ##############################################################################################
    df_sigtib = df[["Subjects", "Tibial SNR", "Sigma_Tibial_Visible"]]
    # print(df_sigtib)
    # exit()
    # df_sigtib_long= df_sigtib.melt(var_name='Method', value_name='SNR (A.U.)')
    dy = "Tibial SNR"
    dx = "Sigma_Tibial_Visible"
    ort = "v"
    pal = sns.color_palette(n_colors=4)
    f, ax = plt.subplots(figsize=(5, 8))
    ax = pt.half_violinplot(x=dx, y=dy, data=df_sigtib, palette=pal, bw=.2, cut=0.,
                            scale="area", width=.6, inner=None, orient=ort,
                            linewidth=0.0)
    ax = sns.stripplot(x=dx, y=dy, data=df_sigtib, palette=pal, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort)
    ax = sns.boxplot(x=dx, y=dy, data=df, color="black", width=.15, zorder=10,
                     showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                     saturation=1, orient=ort)
    # ax = ax.set_ylim[(0, 25)]
    plt.ylim([0, 25])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    result = df.groupby('Sigma_Tibial_Visible')['Tibial SNR'].mean()
    print(result)
    plt.show()
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)
    # print(df)


