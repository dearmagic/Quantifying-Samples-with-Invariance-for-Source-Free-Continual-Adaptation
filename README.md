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


```
python OH_source_Train.py --gpu 0 --source 0
```
## Negative Dataset Generate
- Second, to generate the negative dataset to provide negative sample for CISFDA model
- 

## Adapt to the Target Domain 
- Third, to train CISFDA on the target domain (please assign a source-trained model path):

from Art to Clipart on Office-Home-CI:
```
python -u /public/home/imgbreaker/Desktop/CISFDA/CISFDA/main_test.py \
--source_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/20240429-2246-OH_clipart_ce_singe_gpu_resnet50_best_param.pth \
--weight_model /public/home/imgbreaker/CIUDA/ProCA-main/ProCA-main/model_source/domainNet-126/negative/20240501-1929-OH_clipart_ce_singe_gpu_resnet50_best_neg_param.pth \
--source clipart --target real --model_name final_domainnet --txt clipart2real.txt --dataset domainnet/real \
 > ./log/main_test_clipart2real.log 2>&1 &
```
