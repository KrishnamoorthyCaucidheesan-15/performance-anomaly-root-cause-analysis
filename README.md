# root-cause-analysis-anomaly-detection
Contains code related to root cause analysis for anomaly detection
The code in this repository is to find out the root cause of an identified anomaly.
The anomalies are grouped into four types
1. CPU_hog
2. Backend delay
3. Backend failure
4. User surge

For each use case there are four types of plots derived to find out the root cause
1. Time series plot
2. Metric count histogram
3. Reconstruction error plot
4. Metric level reconstruction error plot

## Repository Structure

### usad.py
This file has the encoder, decoder architecture code and the code for reshaping the model

### utils.py
This file has the code for all four plots 
1. Time series plot
2. Metric count histogram
3. Reconstruction error plot
4. Metric level reconstruction error plot

### root_cause_analysis_code.py
This file has the code for training, testing the USAD model and visualisation code for plotting.

#### The four plots can be viewed [here](RootcauseAnalysisAnomalyDetectionUSAD - Google Drive).

#### The details regarding the root cause analysis anomaly detection using USAD model is explained in this [Document](https://docs.google.com/document/d/1YEydZbjvepgKePz2i2YWppYYOD-p5l60yZ1eX2mzk0U/edit).

#### The required Datasets can be taken [here](https://drive.google.com/drive/u/0/folders/1eHHw3sbE-PVySBjwvlc_6NT0cknQEvHe). 

The usecases have the threshold values independent of eachother. Therefore there is a need for adjustment of percentile values to maintain good precision, recall values.The percentile values for each use case can be viewed [here](https://docs.google.com/spreadsheets/d/1Xb7rOxUV0B5-Ua9AGqrrXdktTXxlDyJwLT2uDKBb_14/edit#gid=0).


