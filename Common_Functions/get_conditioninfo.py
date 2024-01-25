def get_conditioninfo(condition, srmr_nr):
    # return info of the different conditions
    class CondInfo():
        def __init__(self):
            pass

    if srmr_nr == 1:
        # conditions:
        # 1 = rest,
        # 2 = median stimulation,
        # 3 = tibial stimulation,
        # 4 = alternating median and tibial stimulation
        if condition == 1:
            cond_name = 'rest'
            nblocks = 1
            nerve = []
            stimulation = 0
            trigger_name = []
            str_stimulation = cond_name
            str_nerve = []

        elif condition == 2:
            cond_name = 'median'
            nblocks = 4
            nerve = 1
            stimulation = nerve
            trigger_name = 'Median - Stimulation'
            str_nerve = 'med'
            str_stimulation = '_mixed'

        elif condition == 3:
            cond_name = 'tibial'
            nblocks = 4
            nerve = 2
            stimulation = nerve
            trigger_name = 'Tibial - Stimulation'
            str_nerve = 'tib'
            str_stimulation = '_mixed'

        elif condition == 4:
            cond_name = 'alternating'
            nblocks = 2
            nerve = 12
            stimulation = 3
            trigger_name = 'Median - Stimulation' 'Tibial - Stimulation'
            str_nerve = 'alt'
            str_stimulation = '_mixed'

        else:
            print('Error - conditions invalid')

        cond_name2 = cond_name

    elif srmr_nr == 2:
        # conditions:
        # 1 = rest,
        # 2 = median digits,
        # 3 = median mixed nerve,
        # 4 = tibial digits,
        # 5 = tibial mixed nerve

        if condition == 1:
            nblocks = 1
            nerve = 0
            stimulation = 0
            trigger_name = []
            str_stimulation = ''
            str_nerve = 'rest'
            str_cond1 = 'rest'
            str_cond2 = ''

        elif condition == 2:
            nblocks = 4
            nerve = 1
            str_nerve = 'med'
            str_stimulation = '_digits'
            str_cond1 = 'median'
            str_cond2 = 'sensory'

        elif condition == 3:
            nblocks = 1
            nerve = 1
            str_nerve = 'med'
            str_stimulation = '_mixed'
            str_cond1 = 'median'
            str_cond2 = 'mixed'

        elif condition == 4:
            nblocks = 4
            nerve = 2
            str_nerve = 'tib'
            str_stimulation = '_digits'
            str_cond1 = 'tibial'
            str_cond2 = 'sensory'

        elif condition == 5:
            nblocks = 1
            nerve = 2
            str_nerve = 'tib'
            str_stimulation = '_mixed'
            str_cond1 = 'tibial'
            str_cond2 = 'mixed'

        else:
            print('Error - conditions invalid')

        if nerve > 0:
            if str_stimulation == '_mixed':
                stimulation = 1
                trigger_name = str_nerve + 'Mixed'

            elif str_stimulation == '_digits':
                stimulation = 2
                trigger_name = str_nerve + '1', str_nerve + '2', str_nerve + '12'

        cond_name = str_nerve + str_stimulation
        cond_name2 = str_cond1 + str_cond2

    cond_info = CondInfo()
    cond_info.nblocks = nblocks
    cond_info.cond_name = cond_name
    cond_info.nerve = nerve
    cond_info.stimulation = stimulation
    cond_info.trigger_name = trigger_name
    cond_info.str_stimulation = str_stimulation
    cond_info.str_nerve = str_nerve
    cond_info.condition = condition
    cond_info.cond_name2 = cond_name2

    return cond_info
