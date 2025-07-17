[![GitHub Release](https://img.shields.io/github/v/release/eippertlab/hfo-cns)](https://github.com/eippertlab/hfo-cns/releases/tag/v1.0)
[![DOI](https://zenodo.org/badge/554081060.svg)](https://doi.org/10.5281/zenodo.14175917)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# hfo-cns

This repository is associated with the following [preprint](https://www.biorxiv.org/content/10.1101/2024.11.16.622608v1) and the corresponding data in [Dataset 1](https://openneuro.org/datasets/ds004388) 
and [Dataset 2](https://openneuro.org/datasets/ds004389). If you have any questions related to this code, please feel free to 
contact [bailey@cbs.mpg.de](mailto:bailey@cbs.mpg.de).

Abbreviations used:
* CNS: Central nervous system
* HFO: High frequency oscillation
* LF-SEP: Low frequency somatosensory evoked potential


# Content
This repository contains the code used to preprocess and analyse high frequency oscillations 
across the central nervous system in electrophysiology data as presented in the above-mentioned manuscript.

## Main_Processing
**EEG_Pipeline.py** and **ESG_Pipeline.py** are wrapper scripts which can be used to specify the stages of analysis to
run. These steps include:
* Data Import (incl. downsampling, r-peak event annotation, stimulus artefact removal, file concatenation)
* Bad channel and bad trial checks
* Cardiac artefact removal via signal space projection (SSP) with 6 projections (**ESG_Pipeline.py only**)
* Spectral filtering to isolate high frequency oscillations between 400Hz and 800Hz
* Signal enhancement via canonical correlation analysis (CCA)
  * CCA for subcortical and cortical CNS levels is via **EEG_Pipeline.py**
  * CCA for spinal CNS level is via **ESG_Pipeline.py**

## Common_Functions
Scripts contained in [Common_Functions](Common_Functions) contain function files which support the preprocessing and analysis of the 
EEG and ESG data in the aforementioned datasets. They can generally be used independent of the CNS level currently
under study.

## CNS_Level_Specific_Functions
Scripts contained in [CNS_Level_Specific_Functions](CNS_Level_Specific_Functions) contain the functions required to run CCA and subsequently select
the optimal spatial filter (and thus determine the participants with detected HFOs) for each condition at the spinal, subcortical and cortical CNS levels. Additionally, the 
scripts required to run cardiac artefact removal via SSP are found here.

## Group_Level_Analyses
Scripts contained in [Group_Level_Analyses](Group_Level_Analyses) contain scripts required to:
* Obtain the latency of high and low frequency potentials of interest ([GetPotential_Timing_LowFreq_HighFreq.py](GroupLevelAnalyses%2FGetPotential_Timing_LowFreq_HighFreq.py))
* Determine the number of participants with detected HFOs before CCA is applied ([SNR_EnvelopePeak_SensorSpace.py](GroupLevelAnalyses%2FSNR_EnvelopePeak_SensorSpace.py))
* Compute the burst frequency at each CNS level for each participant ([TFR_ROISearch_BurstFrequency_WeightsToBroadband_Filter.py](GroupLevelAnalyses%2FTFR_ROISearch_BurstFrequency_WeightsToBroadband_Filter.py))
  * As well as related statistical analyses ([BurstFreq_AcrossCNS_Stats.py](GroupLevelAnalyses%2FBurstFreq_AcrossCNS_Stats.py))
* Determine the number of burst peaks for each HFO burst ([WaveletCount_EqualWindowCounting.py](GroupLevelAnalyses%2FWaveletCount_EqualWindowCounting.py))
  * As well as related statistical analyses ([WaveletCount_Stats.py](GroupLevelAnalyses%2FWaveletCount_Stats.py))
* Compute the latency, amplitude and signal-to-noise ratio of average HFOs and LF-SEPs ([PartialCorrelation_SNR_CreateTables_CCA.py](GroupLevelAnalyses%2FPartialCorrelation_SNR_CreateTables_CCA.py)), such that we can
  * Compute the correlation and partial correlation between average HFO and LF-SEPs across participants ([PartialCorrelation_LowFreq_HighFreq_CCA_Amp_SNR.py](GroupLevelAnalyses%2FPartialCorrelation_LowFreq_HighFreq_CCA_Amp_SNR.py))
* Compute the single trial signal-to-noise ratio for the HFOs and LF-SEPs for cortical and spinal data ([HighVsLowFreq_ComputeSingleTrial_SNR_CorticalSpinal_CCA.py](GroupLevelAnalyses%2FHighVsLowFreq_ComputeSingleTrial_SNR_CorticalSpinal_CCA.py))
to enable LF-SEP to HFO comparison in the strongest versus weakest trials

## CCA_Resting_State_Validation
In addition to the main analysis, validation of the results obtained via CCA was performed using resting state recordings
which accompany the task-evoked recordings for each participant. 
* [EEG_RestingState_Pipeline.py](EEG_RestingState_Pipeline.py) and [ESG_RestingState_Pipeline.py](ESG_RestingState_Pipeline.py) are 
wrapper scripts used to preprocess the resting state data in a similar fashion to the task-evoked data
* [CCA_Resting_State_Validation](CCA_Resting_State_Validation) contains scripts to perform the validation itself
  * [TaskEvoked_RestingState_CCACorrelation_Parallel.py](CCA_RestingStateValidation%2FTaskEvoked_RestingState_CCACorrelation_Parallel.py) performs
  the validation and computes the correlation between iterations of CCA across the entire trial length
  * [TaskEvoked_RestingState_CCACorrelation_CCAWindow.py](CCA_RestingStateValidation%2FTaskEvoked_RestingState_CCACorrelation_CCAWindow.py) takes the 
  time courses from the previous step and computes the correlation within only the time window used to train CCA
  * [TaskEvoked_RestingState_CCACorrelation_ShorterWindow.py](CCA_RestingStateValidation%2FTaskEvoked_RestingState_CCACorrelation_ShorterWindow.py) takes the 
  time courses from the previous step and computes the correlation from 10ms before to 40ms after the time window used to train CCA
  * [Correlation_SummaryTable_UpperTri.py](CCA_RestingStateValidation%2FCorrelation_SummaryTable_UpperTri.py) computes the average
  absolute correlation across participants
    * [CCA_CorrelationValidation_ttests_UpperTri.py](CCA_RestingStateValidation%2FCCA_CorrelationValidation_ttests_UpperTri.py) performs 
    related statistics

## Noise_Simulations
Supplementary analyses were performed to investigate the impact of filter artefacts (e.g. ringing) on the results of this
study. This directory contains the code required to add scaled 1/f or pink noise to a unit impulse to simulate single-trial
data for different scaling factors that are relevant based on our empirical data.
* [Compute_EvokedAmplitude_Cortical.py](Compute_EvokedAmplitude_Cortical.py) and [Compute_EvokedAmplitude_Spinal.py](Compute_EvokedAmplitude_Spinal.py)
allows for the computation of the peak amplitude of cortical and spinal single-subject evoked responses. The average of these 
peak amplitudes across subjects are used to inform the scaling factor for the noise simulations 
* [PrestimulusNoise_RealSignal.py](PrestimulusNoise_RealSignal.py) and [PrestimulusNoise_RealSignal_HF.py](PrestimulusNoise_RealSignal_HF.py)
allows for the computation of the standard deviation of cortical and spinal single-subject epochs in the baseline period. The average of these 
standard deviations across subjects are used to inform the scaling factor for the noise simulations
* [ImpulseResponse_Filter_Pink.py](ImpulseResponse_Filter_Pink.py) performs the single-trial simulations by summing the unit impulse
and scaled 1/f or pink noise for each of 2000 trials per simulated participant (N=60). These single-trials are then band-pass filtered
from 400-800Hz to investigate the effect of filtering.
* [SNR_NoiseSim.py](SNR_NoiseSim.py) calculates the signal-to-noise ratio of the simulated data

## Publication_Images
Scripts contained in [Publication_Images](Publication_Images) are used to generate all figures presented in the manuscript and supplement.

# Required Software
All scripts run with python 3.10 and MNE 1.7.0, for a detailed list of required packages see [requirements.txt](requirements.txt).
