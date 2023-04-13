# Read in the HFO SNR tables and us df.describe to try and figure out reasonable limits

import os
import mne
import numpy as np
from meet import spatfilt
from Common_Functions.get_conditioninfo import get_conditioninfo
from Common_Functions.calculate_snr import calculate_snr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42


if __name__ == '__main__':
    eeg_path = '/data/p_02718/Images/CCA_eeg/SNR&EnvelopePeak/'
    esg_path = '/data/p_02718/Images/CCA/SNR&EnvelopePeak/'
    esg_good_path = '/data/p_02718/Images/CCA_good/SNR&EnvelopePeak/'

    cond_names = ['Median Stimulation', 'Tibial Stimulation']
    paths = {'Cortical': eeg_path,
             'Spinal': esg_path,
             'Spinal Good Trials Only': esg_good_path
             }

    for cond_name in cond_names:
        for path_key in paths:
            df = pd.read_excel(f'{paths[path_key]}ComponentSNR.xlsx', sheet_name=cond_name, header=0)
            df = df.drop("Unnamed: 0", axis=1)
            print(path_key, cond_name)
            # print(df)
            # print(df.describe())
            print(df.columns)
            cols = [col for col in df.columns]
            # threshold = np.mean(df.mean()) - np.mean(df.std())  # Returns a float
            # print(threshold)
            threshold = df.describe()['Component 1'].loc['25%']
            df_subset = df[(df['Component 1'] > threshold)
                           | (df['Component 2'] > threshold)
                           | (df['Component 3'] > threshold)
                           | (df['Component 3'] > threshold)]
            print(df_subset.count())
