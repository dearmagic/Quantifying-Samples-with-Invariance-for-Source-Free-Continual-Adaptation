# Quantifying-Samples-with-Invariance-for-Source-Free-Continual-Adaptation
This is the official implementation of Quantifying-Samples-with-Invariance-for-Source-Free-Continual-Adaptation.
# Getting Started

- Install the requirements by runing the following command:
```
pip install -r requirements.txt
```

## Data Preparation
- The `.txt` files of data list and its corresponding labels have been put in the directory `./data_splits`.

- Please manually download the Office31, Office-Home and ImageNet-Caltech benchmark from the official websites and put it in the corresponding directory (e.g., '../../dataset/ImageNet-Caltech').

- Put the corresponding `.txt` file in your path (e.g., '../../dataset/ImageNet-Caltech/caltech_list.txt').
## Source Pre-trained
- First, to obtain the pre-trained model on the source domain: 
you need to modified the varible _dataset_ which contains the dataset path and the name of domain.
```
python OH_source_Train.py --gpu 0 --source {index_of_domain(0,1,2,3,...)}
```
## Negative Dataset Generate
- Second, to generate the negative dataset to provide negative sample for CISFDA model
```
python model_perform_valid.py
python negative_dataset_generate.py
```
## Negative Model Training
- Third, you should train negative model
```
python OH_source_Train_negative.py --source {index of domain(0,1,2...)}
  ```
## Adapt to the Target Domain 
- Third, to train CISFDA on the target domain (please assign a source-trained model path):

from Art to Clipart on Office-Home-CI:
```
python main_test.py \
--source_model {your_source_model_path} \
--weight_model {your_negative_model_path} \
--source Art --target Clipart --model_name final_officehome --txt Art2Clipart.txt --dataset office-home/Clipart
```
the parameter _model_name_ only decide the name of model to save.
