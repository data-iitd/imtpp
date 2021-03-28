# IMTPP
Artefacts related to the AISTATS 2021 paper "Learning Temporal Point Processes with Intermittent Observations".

## Requirements
Use a python-3.7 environment and install Tensorflow v1.13.1 and Tensorflow-Probability v0.6.0.

## Execution Instructions
### Dataset Format
Structure the dataset to separated files for train and test event sequences as below.
```
dataset_name/event_train.txt dataset_name/time_train.txt dataset_name/event_test.txt dataset_name/time_test.txt
```
We have also provided a synthetic dataset as a reference.

### Running the Code
Use the following command to run IMTPP on the dataset.
```
python run.py dataset_name
```

## Citing
If you use this code in your research, please cite:
```
@inproceedings{gupta21,
 author = {Vinayak Gupta and Abir De and Sourangshu Bhattacharya and Srikanta Bedathur},
 title = {Learning Temporal Point Processes with Intermittent Observations},
 booktitle = {Proc. of the 24th International Conference on Artificial Intelligence and Statistics (AISTATS)},
 year = {2021}
}
```

## Contact
In case of any issues, please send a mail to
```guptavinayak51 (at) gmail (dot) com```