# An-Investigation-on-the-Performance-of-Domain-Adaptation-for-Code-Completion

This repogitory is the replication package for our study (An Investigation on the Performance of Domain Adaptation for Code Completion)

## Dependencies
wandb                        0.12.6  
tokenizers                   0.10.3  
torch                        1.10.0+cu113  
transformers                 4.12.3  
javalang           0.13.0  
lizard             1.17.10  
nltk               3.7  
tqdm               4.64.0  

## How to Replicate our study
### 1. Prepare a pretrained RoBERTa model and a tokenizer
Prepare a pre-trained model following [this study](https://github.com/RoBERTaCode/roberta), which is our baseline.

### 2. Fine-tune the pre-trained model. 
To fine tune a model, execute the following commands:
```
$ python code/run_fune-tuning.py --model [path of the pre-trained model] --train_data_file [path of train data file (i.g. training_masked_code.txt)] --eval_data_file [path of evaluation data file (i.g. eval_masked_code.txt)] --output_root [directory to output the fine-tuned model] --tokenizer_name [path of tokenizer] --vocab_size [vocab_size]
```

### 3. Evaluate the fine-tuned model.  (WIP)
To evaluate the peformance of the model, execute the following commands:
```
$ python code/~
```

### ex. We provide code to create detaset for a specific Java repogitory or Java corpus.  (WIP)
To create datase for a specific Java repogitory, first prepare the target repogitory (i.g. git clone)

Then, execute the following commands:
```
$ python code/~~
```
