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
This file has the code for training, testing USAD model and visualisation code for plotting.

### outputs
This folder has the output plots for the four use cases
1. CPU_hog
2. Backend delay
3. Backend failure
4. User surge



