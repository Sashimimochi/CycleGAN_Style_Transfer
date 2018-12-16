# CycleGAN on Text Style Transfer
  Unsupervised learning style transfer from negative to positive and vice versa.  

## Prerequisites

1. Python 2.7 or higher
2. tensorflow-gpu 1.0.0 or 1.1.0

## Clone this repository
`git clone https://github.com/adfsghjalison/CycleGAN_Style_Chatbot.git`


## Usage

### Data
`mkdir data`  
`mkdir data/data_[database_name]`  
1. Put training data `source_train` and testing data `source_test` in `data/data_[database_name]`  
format : one data a line  
[style label] +++$+++ [sentence]

2. Put the word dictionary `dict` in `data/data_[database_name]`  
`dict` : a json file with  
`word : word_id`  
with `__BOS__`, `__EOS__`, `__UNK__`  

3. Put the word file `word` in `data/data_[database_name]`   
format: one word a line  
these words would be trained in language model  

### Train word to vector
`python word2vec.py`

### Pretrain Generator
`python main.py --mode pretrain`

### Train
`python main.py --mode train`

### Test
`python main.py --mode test`

### Important Hyperparameters of the flags.py
`batch_size` : batch size  
`sequence_length` : max length of input and output sentence  
`id_loss` : Use identity loss or not  
`dis_it` : discriminator training iterations  
`gen_it` : generator training iterations  

## Files

### Folders
`data/` : training data / testing data / dictionary file / word file  
`model/` : saved trained models  

### Files
`word2vec.py` : train word to vector  
`flags.py` : all settings  
`utils.py` : data processing functions  
`cycle_gan.py` : model architecture  
`main.py` : main function  


