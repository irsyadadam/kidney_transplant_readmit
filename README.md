# Organ Transplant Readmission Prediction

[WRITE DESCR]

**Dataset** 

[WRITE DESCR]

- Number of Classes: 
- Class Distribution: 
- Number of Dimensions: 

[WRITE PREPROCESSING]

# 

### Install Env:

    conda create -n pyg_CUDA python=3.11
    bash INSTALL_ENV.sh


Dependencies: 
- python 3.11
- cuda 11.8
- torch 2.2.1
- torchvision 0.17.1
- torchaudio  2.2.1
- torch_geometric 2.5.0

#

### Model Architecture:

    modality1 --> channel 1 --> x     | 
            | 
            |--> channel 2  --> x     | 
                                       --- (concat) --> (avg time series) 
            |--> channel 3  --> x     | 
            | 
    modality2 --> channel 4 --> x     | 

Remarks: 

 - [WRITE DESCR]
