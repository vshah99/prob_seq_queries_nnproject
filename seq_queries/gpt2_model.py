#################################################################################
#
#             Project Title:  GPT-2 Modeling
#             Author:         Sam Showalter
#             Date:           2022-05-02
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy

import torch
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_dataset

#################################################################################
#   Function-Class Declaration
#################################################################################

def prep_gpt2(batch_size = 128,
              **kwargs):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    dataset = load_dataset("bookcorpus", split="validation[:1%]")
    dataset = dataset.map(lambda e: tokenizer(e['sentence1'], truncation=True, padding='max_length'), batched=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return {
        "model":model,
        "tokenizer": tokenizer,
        "dataset":dataset,
    }

res = prep_gpt2()
print("HI")


#################################################################################
#   Main Method
#################################################################################



