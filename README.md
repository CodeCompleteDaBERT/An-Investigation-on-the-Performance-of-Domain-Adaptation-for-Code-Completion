# An-Empirical-Investigation-on-the-Performance-of-Domain-Adaptation-for-ML-based-Code-Completion

This repogitory is the replication package for our study (An Empirical Investigation on the Performance of Domain Adaptation for ML-based Code Completion)

## Dependencies

wandb 0.12.6  
tokenizers 0.10.3  
torch 1.10.0+cu113  
transformers 4.12.3  
javalang 0.13.0  
lizard 1.17.10  
nltk 3.7  
tqdm 4.64.0

## How to run

### 1. Prepare a pretrained RoBERTa model and a tokenizer

Prepare a pre-trained model and tokenizer following [this study](https://github.com/RoBERTaCode/roberta), which is our baseline.

### 2. Fine-tune the pre-trained model.

To fine tune a model, execute the following commands:

```
$ python code/run_fune-tuning.py --model [path of the pre-trained model] --train_data_file [path of a training dataset file (i.g. training_masked_code.txt)] --eval_data_file [path of a evaluation dataset file (i.g. eval_masked_code.txt)] --output_root [directory to output the fine-tuned model] --tokenizer_name [path of tokenizer] --vocab_size [vocab_size]
```

The dataset for fine-tuning are provided in ./DatasetForDARoBERTa and ./DatasetForRoBERTa+.

### 3. Evaluate the fine-tuned model. 

To evaluate the peformance of the model, execute the following commands:

```
$ python code/evaluate.py --model [path of the pre-trained model] --test_data_file [path of a test dataset file] --output [directory to output test results]
```

### ex. We also provide code to create detaset for a specific Java repogitory or Java corpus. (WIP)

To create datase for a specific Java repogitory, first prepare the target repogitory (i.g. git clone)

Then, execute the following commands to make dataset:

```
$ python code/1_java_path.py --repo[path to repogitory from which you want to make dataset]
$ python code/2_java_filtering.py
$ python code/3_method_splitting.py
$ python code/4_java_masking.py
```
