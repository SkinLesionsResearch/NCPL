# NCPL system
Trying to handle ham10000 dataset with semi-supervised learning methods for skin lesion image classification.

## About:
NCPL (Noisy-consistent Pseudo Labeling Model) provides a noisy consistent semi-supervised framework. In the absence of clear annotation image samples, this approach incorporates a robust pseudo label creation and selection method, as well as an attentive feature integration component, to categorize skin lesion images.

## Installation:

Installation Tested on Ubuntu 20.04, Python >= 3.8, on one NVIDIA RTX 3080Ti GPU.

run `pip3 install -r requirements.txt` to install all the dependencies.


## Data Preparation:
Downloads the HAM10000 dataset and store it in the folder "data/semi_processed"

- Data Path:

The labeled training data subdirectory is the "data/semi_processed/[Num of Samples]/train_labeled.txt", the Num_of_Samples is set at [500, 1000, 1500, 2000, 2500]

The unlabeled training data subdirectory is the "data/semi_processed/[Num of Samples]/train_unlabeled.txt", the Num_of_Samples is set at [500, 1000, 1500, 2000, 2500]

The testing data subdirectory is the "data/semi_processed/test.txt"

- File Format: The txt file for both training and testing data is as following format

"./../ups/data_ham10000/datasets/ISIC_0030828.jpg 4",

The first "./../ups/data_ham10000/datasets/ISIC_0030828.jpg" is the path to image on dist,

The second "4" is the corresponding label.

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
![image](https://github.com/SkinLesionsResearch/NCPL/blob/master/IMG/PLGS.png)
## Results
Ready to publish afterwards.
