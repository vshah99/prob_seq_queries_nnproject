#################################################################################
#
#             Project Title:  GPT-2 Modeling
#             Date:           2022-05-02
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
import types
import pandas as pd

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset

from .utils import read_pkl, write_pkl, write_json, _tup_cpu, _tup_gpu_gpt2, _tup_cpu_gpt2

#################################################################################
#   Function-Class Declaration
#################################################################################


class Gpt2ClassificationCollator(object):

    def __init__(self, use_tokenizer,
                 max_sequence_len=None,
                 min_seq_len=20):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        self.min_seq_len = min_seq_len
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

    def __call__(self, sequences):

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences
                 if len(sequence['text']) > self.min_seq_len]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt",
                                    padding=True, truncation=True,
                                    max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        # inputs.update({'labels':torch.tensor(labels)})
        keep_row = (inputs['attention_mask'].sum(dim=-1) >= self.min_seq_len)
        inputs['input_ids'] = inputs['input_ids'][keep_row,:self.min_seq_len].to('cuda:4')
        inputs['attention_mask'] = inputs['attention_mask'][keep_row,:self.min_seq_len].to('cuda:4')

        return inputs

#######################################################################
# GPT2 Causal query LM
#######################################################################


def load_GPT2_query_lm(device):
    # May be able to use a smaller model
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
    print(total_params)
    model.model_iters = 0
    model.temperature = None

    @torch.no_grad()
    def get_next_probs(self,
        x, rnn_args=None,
        temperature=1.0,
        max_batch_size=16, device='cpu',
        return_forward_only=False,
        return_logits=True, **kwargs
 ):
        hidden_states = rnn_args
        self.model_iters += x.shape[0] * x.shape[1]
        if self.temperature is not None:
            temperature = self.temperature
        xs = torch.split(x,max_batch_size)
        if hidden_states is not None:
            if isinstance(hidden_states,tuple):
                # Need to split up hidden states
                # (num_splits, (num_layers, (2)))
                hidden_states = list(zip(*[
                    zip(*(torch.split(h[0],max_batch_size), torch.split(h[1],max_batch_size)))
                     for h in hidden_states]))
            else: hidden_states = torch.split(hidden_states,max_batch_size)
        else: hidden_states = [None]*len(xs)

        prob_outputs = []; step_outputs = []
        for x, hidden_state in zip(xs, hidden_states):
            hidden_state = _tup_gpu_gpt2(hidden_state,device)
            step_output = self.forward(
                input_ids=x.to(device),
                past_key_values=hidden_state,
                use_cache=True,
                return_dict=True,
            )
            if not return_forward_only:
                logits = step_output["logits"][:, -1, :] / temperature # last position in the sequence
            else: logits = step_output['logits']/temperature
            if not return_logits:
                probs = torch.softmax(logits, dim=-1)
                prob_outputs.append(probs.cpu())
            else:
                prob_outputs.append(logits.cpu())
            step_outputs.append(_tup_cpu_gpt2(step_output['past_key_values']))

        layer_hiddens = []
        for layer_data in zip(*step_outputs):
            # (num_layer, (2))
            # Take manageable input and feed it through this with different batch sizes
            # Then, compare the outputs
            layer_data= list(zip(*layer_data))
            layer_hiddens.append(
                (torch.cat(layer_data[0],dim=0).cpu(),
                 torch.cat(layer_data[1],dim=0).cpu())
            )

        return torch.cat(prob_outputs,dim = 0), tuple(layer_hiddens)

    model.get_next_probs = types.MethodType(get_next_probs, model)
    return model.to(device)



def explore_gpt2(batch_size = 16,
              device=0,
              **kwargs):
    """
    Test functions for GPT2
    """
    samples = [
        "in my opinion, ",
        "to clarify ",
        "go",
        "hi, my name is ",
        "where is ",
        "I don't ",
        "get",
     ]
    res = ['?','.','!',';']

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    data = []
    for s in samples:
        # print(s,tokenizer.encode(s))
        data.append(tokenizer.encode(s))
    return data

