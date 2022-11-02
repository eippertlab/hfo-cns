def get_esg_channels():
    # returns ESG channel names separated in cervical and lumbar channels
    brainstem_chans = ['Iz', 'SC1', 'Fz-TH6']

    cervical_chans = ['S3', 'S5', 'S7', 'S4', 'S6', 'S8', 'S9', 'S11', 'SC6',
                      'S13', 'S15', 'S12', 'S14', 'S16', 'S17', 'S19', 'S18', 'AC']

    lumbar_chans = ['S20', 'S22', 'S24', 'S21', 'S23', 'S25', 'S26', 'S28', 'L1',
                    'S30', 'S32', 'S29', 'S31', 'S33', 'S34', 'S36', 'S35', 'AL', 'L4']

    ref_chan = ['TH6']  # recording reference

    return brainstem_chans, cervical_chans, lumbar_chans, ref_chan
