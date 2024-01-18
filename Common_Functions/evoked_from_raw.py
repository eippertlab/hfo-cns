# Create an evoked response from a raw input

import mne


# Create epochs and evoked response
def evoked_from_raw(raw, iv_epoch, iv_baseline, trigger_name, reduced_epochs):
    events, event_ids = mne.events_from_annotations(raw)
    event_id_dict = {key: value for key, value in event_ids.items() if key == trigger_name}
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=iv_epoch[0], tmax=iv_epoch[1],
                        baseline=tuple(iv_baseline))

    if reduced_epochs and trigger_name == 'Median - Stimulation':
        epochs = epochs[900:1100]
    elif reduced_epochs and trigger_name == 'Tibial - Stimulation':
        epochs = epochs[800:1200]

    evoked = epochs.average()  # This line as is drops the ECG channel
    # evoked = epochs.average(picks='all')  # Keeps ECG channel

    return evoked
