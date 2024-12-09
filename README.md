# cnn2id-based NIDS on the CICIDS2017 and CICIDS Dataset

### Dataset from above

https://www.unb.ca/cic/datasets/ids-2017.html
https://www.unb.ca/cic/datasets/ids-2018.html

## Introduction

This is the source code for the paper entitled "An Intrusion Detection System via novel Convolutional Neural Network"

In this repository, we propose  CNN2ID (Convolutional Neural Network for Intrusion Detection), an intrusion detection algorithm based on the CNN architecture. In extensive experiments, we achieved a high accuracy of up to 99% with CNN2ID, while requiring reasonable training time. CNN2ID achieves state-of-the-art detection performance and outperforms traditional methods in intrusion detection. 

## Installation

Use conda environment

# This file may be used to create an environment using:

$ conda create --name <env> --file <this file>

platform: linux-64

<this file> is requirements.txt

## Requirements

All the experiments were conducted using a 64-bit Intel(R) Core(TM) i7 CPU with 32GB RAM in Linux environment. The models have been implemented in Python v3.7.12 using the PyTorch v1.9.0 library.

## Data

We use the 'CIC-IDS-2017\MachineLearningCSV' and 'CSE-CIC-IDS2018\Processed Traffic Data for ML Algorithms' download from the above link.
for example:
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv\\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv

## How to use

- Install Conda env like the requirements.

- Download dataset and use csv file, copy the csv file into the data folder.

- Run jupyter notebook in the notebook folder step by step.

- The experiment pictures are saved in the image folder.

## References

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Authors

## Citation
