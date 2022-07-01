

import argparse
import glob
import logging
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedModel,
    AutoModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

import time
import sys
import os
import wandb
import random

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from attrdict import AttrDict

logger = logging.getLogger("evaluate")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = AttrDict({
"output_dir":"./evaluate/",
"local_rank":0,
"eval_batch_size":1,
"eval_data_file":"./complete/test_masked_code.txt",
"device":device,
"model_path":"./output/glamorous-sweep-1",
"tokenizer_name":"./output/leafy-sweep-1/",
"predict_result":"./DATA/predict_spring.txt"
})



class LineByLineDatasetWithBPETokenizer(Dataset):
    def __init__(self, file_path: str = None, tokenizer_path: str = None):
        tokenizer = ByteLevelBPETokenizer(
            tokenizer_path + "/vocab.json",
            tokenizer_path + "/merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line for line in lines if (len(line) > 0 and not line.isspace())]
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


def load_and_cache_examples(args, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return LineByLineDatasetWithBPETokenizer(file_path, args.tokenizer_name)

def evaluate(args, model, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )


    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    labels_file = str(args.eval_data_file).replace('masked_code', 'mask')
    labels_lines = [line.rstrip() for line in open(labels_file)]

    step = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # Get the labels lines to process
        start = step * len(batch)
        end = start + len(batch) + 1
        lables_to_process = labels_lines[start:end]

        step += 1

        inputs, labels = read_masked_dataset(tokenizer, batch, lables_to_process)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    perfect_predictions, num_examples = get_number_perfect_predictions(model, tokenizer, args.eval_data_file)
    result = {'perfect_predictions_percentage': perfect_predictions / num_examples,'test': args.eval_data_file,'model': args.model_path,"perplexity": perplexity, "loss": eval_loss,
              "perfect_predictions": perfect_predictions, "total_eval_examples": num_examples}

    logger.info({'val_perplexity': perplexity, 'avg_val_loss': eval_loss})
    logger.info({'perfect_predictions': perfect_predictions})
    logger.info({'perfect_predictions_percentage': perfect_predictions / num_examples})
    logger.info({'model': args.model_path})
    logger.info({'test': args.eval_data_file})

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_" + str(time.time()) + ".txt")
    with open(output_eval_file, "w") as writer:
        logging = logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
    return result

def read_masked_dataset(tokenizer: PreTrainedTokenizer, batch, labels_to_process) -> Tuple[torch.Tensor, torch.Tensor]:
    # The inputs are already masked in the training file
    tmp_inputs = batch.clone()

    tmp_inputs_list = []
    for input in tmp_inputs:
        decoded_input = tokenizer.decode(input)
        encoded_back = tokenizer.encode(decoded_input)[1:-1] # Removes the additional <s> and </s> added
        tmp_inputs_list.append(encoded_back)

    # Gets the maximum length between inputs and labels_lines
    # We then need to adapt one or the other to have the same length through padding
    max_length_inputs = max([len(input) for input in tmp_inputs_list])
    max_length_labels_lines = max([len(label) for label in labels_to_process])
    max_length = max_length_inputs
    if max_length_labels_lines > max_length_inputs:
        max_length = max_length_labels_lines

    # Create the labels tensor
    labels_to_convert_in_tensor = []

    i = 0
    while i < len(batch):
        l1_tmp = tokenizer.encode(labels_to_process[i])
        label_to_add = []
        for token in l1_tmp:
            if token != tokenizer.bos_token_id and token != tokenizer.eos_token_id:  # Remove special tokens
                label_to_add.append(token)

        j = len(label_to_add)
        while j < max_length:
            label_to_add.append(-100)  # we only compute loss for masked tokens
            j += 1

        labels_to_convert_in_tensor.append(label_to_add)
        i += 1

    labels = torch.as_tensor(labels_to_convert_in_tensor)

    inputs_to_convert = []
    for input in tmp_inputs_list:
        tmp_input = []
        for token in input:
            tmp_input.append(token)

        i = len(tmp_input)
        while i < max_length:
            tmp_input.append(tokenizer.pad_token_id)
            i += 1
        inputs_to_convert.append(tmp_input)

    inputs = torch.as_tensor(inputs_to_convert)

    return inputs, labels


def get_number_perfect_predictions(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_data_file):
    labels_file = str(eval_data_file).replace('masked_code', 'mask')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inputs
    with open(eval_data_file) as f:
        inputs = f.readlines()
    inputs = [x.strip() for x in inputs]

    # Targets
    with open(labels_file) as f:
        targets = f.readlines()
    targets = [x.strip() for x in targets]

    n_perfect_predictions = 0
    i = 0
    
    #保存
    with open(args.predict_result, mode='w') as f:
        predict_f = f
        while i < len(inputs):
            input = inputs[i]
            target = "".join(targets[i].split()).replace('<z>', '')

            indexed_tokens = tokenizer.encode(input)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(device)
            with torch.no_grad():
                outputs = model(tokens_tensor)
                predictions = outputs[0]

            predicted_sentence = []
            for token in torch.argmax(predictions[0], 1).cpu().numpy():
                if token != tokenizer.convert_tokens_to_ids('<z>'):
                    predicted_sentence.append(token)
                else:
                    break

            prediction = tokenizer.decode(predicted_sentence)
            prediction = "".join(prediction.split())
            predict_f.write(prediction+"\n")
            
            if target == prediction:
                n_perfect_predictions += 1
            i += 1

    return n_perfect_predictions, len(inputs)


test_data = ["./complete/testing_masked_code.txt"]
model_paths = ["./0512/super-sweep-1"] 
for test in test_data:
    for model_path in model_paths:
        args.eval_data_file = test
        args.model_path = model_path
        try:  
            model = RobertaForMaskedLM.from_pretrained(args.model_path)
            tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
            model.to(device)
            evaluate(args, model, tokenizer)
        except Exception:
            pass