# NCPL system
Trying to handle ham10000 dataset with semi-supervised learning methods for skin lesion image classification.

## About:
NCPL (Noisy-consistent Pseudo Labeling Model) provides a consistency semi-supervised framework. This framework incoroprates a robust pseudo label 
generation and selection mechanism and an attentive feature integration components to classify skin lesion images in a scenario which is lack of 
clean annotation image samples.

## Installation:

Installation Tested on Ubuntu 20.04, Python >= 3.8, on one NVIDIA RTX 3080Ti GPU.

run `pip3 install -r requirements.txt` to install all the dependencies.


## Data Preparation:
Downloads the HAM10000 dataset and store it in the folder "data/semi_processed"

- Data Path:

The labeled training data subdirectory is the "data/semi_processed/[Num of Samples]/train_labeled.txt", the Num_of_Samples is set at [500, 1000, 1500, 2000, 2500]

The unlabeled training data subdirectory is the "data/semi_processed/[Num of Samples]/train_unlabeled.txt", the Num_of_Samples is set at [500, 1000, 1500, 2000, 2500]

The testing data subdirectory is the "data/semi_processed/test.txt"

## Training
```shell

# For classification on 7 Labels

python -u ./train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1500 \
--num_classes 7 \
--batch_size 32

# For classification on 2 Labels

python -u ./train.py \
--src-dset-path './data/semi_processed' \
--labeled_num 1500 \
--num_classes 7 \
--batch_size 32

# Monitoring training progress
tensorboard --logdir=results
```
## System Architecture
### PLGS Module
![image](https://github.com/SkinLesionsResearch/NCPL/blob/ncpl_demo/ArchitectureGraph/PLGS.png)
### ACFI Module
![image](https://github.com/SkinLesionsResearch/NCPL/blob/ncpl_demo/ArchitectureGraph/ACFI.png)
