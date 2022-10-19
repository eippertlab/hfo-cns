def get_channels(subject_nr, includesEcg, includesEog, study_nr):
    # return eeg_chans, esg_chans, bipolar_chans

    # returns EEG, ESG and bipolar channel names
    esg_chans1 = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                  'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                  'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                  'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                  'S23']
    esg_chans2 = ['S35', 'S24', 'S36', 'Iz', 'S17', 'S15', 'S32', 'S22',
                  'S19', 'S26', 'S28', 'S9', 'S13', 'S11', 'S7', 'SC1', 'S4', 'S18',
                  'S8', 'S31', 'SC6', 'S12', 'S16', 'S5', 'S30', 'S20', 'S34', 'AC',
                  'S21', 'S25', 'L1', 'S29', 'S14', 'S33', 'S3', 'AL', 'L4', 'S6',
                  'S23', 'Fz-TH6']
    #includes Fz

    bipolar_chans1 = ['BreathBelt', 'ECG', 'EP', 'Biceps', 'Thumb',
                      'Toe', 'KneeM', 'Knee1', 'Knee2', 'Knee3', 'Knee4']

    bipolar_chans2 = ['BreathBelt', 'ECG', 'EP', 'Biceps',
                      'KneeM', 'Knee1', 'Knee2', 'Knee3', 'Knee4']
    #without EMG at toe and thumb

    eeg_chans1 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1',
                  'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'AFz', 'Fz', 'Cz', 'Pz',
                  'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'FT9', 'FT10',
                  'LM', 'FCz', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3',
                  'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6',
                  'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FPz', 'CPz',
                  'F9', 'F10']
    eeg_chans2 = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                  'T7', 'T8', 'P7', 'P8', 'AFz', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5',
                  'FC6', 'CP5', 'CP6', 'LM', 'FCz', 'C1', 'C2', 'FC3', 'FC4', 'CP3', 'CP4', 'C5', 'C6',
                  'CPz']

    # define electrode setup for the two experiments:
    if study_nr == 1:
        if subject_nr < 7:
            esg_chans = esg_chans1
        else:
            esg_chans = esg_chans2
        eeg_chans = eeg_chans1
        bipolar_chans = bipolar_chans1

    elif study_nr == 2:
        esg_chans = esg_chans2
        eeg_chans = eeg_chans2
        bipolar_chans = bipolar_chans2

    # include ECG
    if includesEcg:
        eeg_chans.append('ECG')
        esg_chans.append('ECG')

    # include EOG
    if includesEog:
        if study_nr == 2:
            eeg_chans.append('EOGH')
            eeg_chans.append('EOGV')

    return eeg_chans, esg_chans, bipolar_chans
