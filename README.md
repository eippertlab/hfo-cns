# cns-hfo

This repository is associated with the following [manuscript] and the corresponding [dataset1](https://openneuro.org/datasets/ds004388) 
and [dataset2](https://openneuro.org/datasets/ds004389). If you have any questions related to this code, please feel free to 
contact [bailey@cbs.mpg.de](mailto:bailey@cbs.mpg.de).

Abbreviations:
* CNS: Central nervous system
* HFO: High frequency oscillation
* LF-SEP: Low frequency somatosensory evoked potential


# Content
This repository contains the preprocessing and analysis code used to preprocess and analyse high frequency oscillations 
across the central nervous system in electrophysiology data as presented in the above-mentioned manuscript.

## Main Processing
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

## CNSLevelSpecificFunctions
Scripts contained in [CNSLevelSpecificFunctions](CNSLevelSpecificFunctions) contain the functions required to run CCA and subsequently select
the optimal spatial filter for each condition at the spinal, subcortical and cortical CNS levels. Additionally, the 
scripts required to run cardiac artefact removal via SSP are found here.

## GroupLevelAnalyses
Scripts contained in [GroupLevelAnalyses](GroupLevelAnalyses) contain scripts required to:
* Compute the burst frequency at each CNS level for each participant ([TFR_ROISearch_BurstFrequency_WeightsToBroadband_Filter.py](GroupLevelAnalyses%2FTFR_ROISearch_BurstFrequency_WeightsToBroadband_Filter.py) and [TFR_ROISearch_BurstFrequency_WeightsToBroadband_Filter_Digits.py](GroupLevelAnalyses%2FTFR_ROISearch_BurstFrequency_WeightsToBroadband_Filter_Digits.py))
  * As well as related statistical analyses ([BurstFreq_AcrossCNS_Stats.py](GroupLevelAnalyses%2FBurstFreq_AcrossCNS_Stats.py))
* Obtain the latency of high and low frequency potentials of interest ([GetPotential_Timing_LowFreq_HighFreq.py](GroupLevelAnalyses%2FGetPotential_Timing_LowFreq_HighFreq.py))
* Compute the single trial signal-to-noise ratio for the HFOs and LF-SEPs for cortical and spinal data ([HighVsLowFreq_ComputeSingleTrial_SNR_CorticalSpinal_CCA.py](GroupLevelAnalyses%2FHighVsLowFreq_ComputeSingleTrial_SNR_CorticalSpinal_CCA.py))
* Compute the latency, amplitude and signal-to-noise ratio of HFOs and LF-SEPs ([PartialCorrelation_SNR_CreateTables_CCA.py](GroupLevelAnalyses%2FPartialCorrelation_SNR_CreateTables_CCA.py)), such that we can
  * Compute the correlation and partial correlation between grand average HFO and LF-SEPs across participants ([PartialCorrelation_LowFreq_HighFreq_CCA_Amp_SNR.py](GroupLevelAnalyses%2FPartialCorrelation_LowFreq_HighFreq_CCA_Amp_SNR.py))
* Determine the number of participants with detected HFOs before CCA is applied ([SNR_EnvelopePeak_SensorSpace.py](GroupLevelAnalyses%2FSNR_EnvelopePeak_SensorSpace.py))
* Determine the number of burst peaks for each HFO burst ([WaveletCount_EqualWindowCounting.py](GroupLevelAnalyses%2FWaveletCount_EqualWindowCounting.py))
  * As well as related statistical analyses ([WaveletCount_Stats.py](GroupLevelAnalyses%2FWaveletCount_Stats.py))

## Publication_Images
Scripts contained in [Publication_Images](Publication_Images) are used to generate all figures presented in the manuscript and supplement.

# CCA validation using resting state recordings
In addition to the main analysis, validation of the results obtained via CCA was performed using resting state recordings
which accompany the task-evoked recordings for each participant. 
* [EEG_RestingState_Pipeline.py](EEG_RestingState_Pipeline.py) and [ESG_RestingState_Pipeline.py](ESG_RestingState_Pipeline.py) are 
wrapper scripts used to preprocess the resting state data in a similar fashion to the task-evoked data
* [CCA_RestingStateValidation](CCA_RestingStateValidation) contains scripts to perform the validation itself
  * [TaskEvoked_RestingState_CCACorrelation_Parallel.py](CCA_RestingStateValidation%2FTaskEvoked_RestingState_CCACorrelation_Parallel.py) performs
  the validation and computes the correlation between iterations of CCA across the entire trial length
  * [TaskEvoked_RestingState_CCACorrelation_CCAWindow.py](CCA_RestingStateValidation%2FTaskEvoked_RestingState_CCACorrelation_CCAWindow.py) takes the 
  time courses from the previous step and computes the correlation within only the time window used to train CCA
  * [TaskEvoked_RestingState_CCACorrelation_ShorterWindow.py](CCA_RestingStateValidation%2FTaskEvoked_RestingState_CCACorrelation_ShorterWindow.py) takes the 
  time courses from the previous step and computes the correlation within only from 10ms before to 40ms after the time window used to train CCA
  * [Correlation_SummaryTable_UpperTri.py](CCA_RestingStateValidation%2FCorrelation_SummaryTable_UpperTri.py) computes the average
  absolute correlation across participants
    * [CCA_CorrelationValidation_ttests_UpperTri.py](CCA_RestingStateValidation%2FCCA_CorrelationValidation_ttests_UpperTri.py) performs 
    related statistics

# Required Software
All scripts run with python 3.9 and MNE 1.0.3,  for an extensive list of required packages see [requirements.txt](requirements.txt).
