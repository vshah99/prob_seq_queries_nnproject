#################################################################################
#
#             Project Title:  Sampling
#             Author:         Sam Showalter and Alex Boyd
#             Date:           2022.04.13
#
#################################################################################

#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
import pickle as pkl

import numpy as np
import random
import torch
import torch.nn as nn

from tqdm import tqdm
from .data import load_text, process_data
from .model import CausalLM, MaskedLM
from .utils import top_k_top_p_filtering

#################################################################################
#   Function-Class Declaration
#################################################################################


def uniform_proposal(hists, sample_len, model, vocab_size, excluded_terms,
                     device='cpu', **kwargs):
    assert(len(hists.shape) == 2)

    # Uniformly sample across the restricted vocabulary indices
    samples = torch.randint(low=0, high=vocab_size-len(excluded_terms), size=(hists.shape[0], sample_len), device=hists.device)
    for item in sorted(excluded_terms):
        samples[samples>=item] += 1
    assert(samples.max() < vocab_size)

    output = model.forward(src=torch.cat((hists, samples), dim=-1))  #, device=device)
    logits = output["logits"]
    model_log_prob = torch.log_softmax(logits, dim=-1)[..., -(sample_len+1):-1, :]
    model_log_prob = torch.gather(model_log_prob, dim=-1, index=samples.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # grab specific log probabilities

    return {
        "proposal_log_prob": -sample_len * np.log(vocab_size - len(excluded_terms)),
        "model_log_prob": model_log_prob.unsqueeze(-1),
        "samples": samples,
        "next_log_dist": torch.log_softmax(logits, dim=-1)[..., -1, :],
    }

def lm_proposal(hists, sample_len, model, vocab_size, excluded_terms,
                device='cpu',top_k=0, top_p=1.0, temperature=1.0,  **kwargs):
    assert(len(hists.shape) == 2)

    proposal_log_prob, model_log_prob = 0.0, 0.0
    samples = []
    last_sample, rnn_args = hists, None
    for _ in range(sample_len):
        logits, rnn_args = model.get_next_probs(last_sample, rnn_args=rnn_args, device=device, return_logits=True)

        proposal_logits = logits.clone()
        proposal_logits[..., excluded_terms] = -float('inf')
        proposal_logits = torch.log_softmax(top_k_top_p_filtering(proposal_logits/temperature, top_k=top_k, top_p=top_p), dim=-1)

        last_sample = torch.distributions.Categorical(logits=proposal_logits).sample().unsqueeze(-1)
        proposal_log_prob += torch.gather(proposal_logits, dim=-1, index=last_sample).squeeze(-1)
        model_log_prob += torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=last_sample).squeeze(-1)
        samples.append(last_sample)

    logits, _ = model.get_next_probs(last_sample, rnn_args=rnn_args, device=device, return_logits=True)  # get last subsequent distribution

    samples = torch.cat(samples, dim=-1)

    return {
        "proposal_log_prob": proposal_log_prob.unsqueeze(-1),
        "model_log_prob": model_log_prob.unsqueeze(-1),
        "samples": samples,
        "next_log_dist": torch.log_softmax(logits, dim=-1),
    }

@torch.no_grad()
def mc_estimate(hist, num_mc_samples, sample_len, model, excluded_terms, proposal_func,
                vocab_size, batch_size=128,temperature=1, top_k=None, top_p=None, device='cpu',**kwargs):
    assert(len(hist.shape) == 1)  # (hist_seq_len), Only conditions on a single history
    dist_estimate = 0
    remaining_samples = num_mc_samples
    while remaining_samples > 0:
        sample_out = proposal_func(
            hists=hist.unsqueeze(0).expand(min(remaining_samples, batch_size), -1),
            sample_len=sample_len,
            model=model,
            vocab_size=vocab_size,
            excluded_terms=excluded_terms,
            top_k=top_k,
            top_p=top_p,
            device=device,
            temperature=temperature,
        )
        remaining_samples -= batch_size
        term_log_prob = sample_out["next_log_dist"] + sample_out["model_log_prob"] - sample_out["proposal_log_prob"]
        dist_estimate += term_log_prob.exp().sum(dim=0) / num_mc_samples

    return dist_estimate

def geom_interp(target_pct, n_current, n_end):
    return target_pct ** ((n_current + 1) / n_end)  # add 1 due to 0-indexing

def lin_interp(target_pct, n_current, n_end):
    a = geom_interp(target_pct=target_pct, n_current=0, n_end=n_end)
    b = target_pct
    t = n_current / (n_end - 1)
    return a * (1 - t) + b * t

@torch.no_grad()
def beam_search_lower_bound(hist, num_beams, sample_len, model, excluded_terms, interp_func,
                            device, vocab_size, **kwargs):
    assert(isinstance(num_beams, (int, float)))
    assert(len(hist.shape) == 1)

    beams, rnn_args = hist.unsqueeze(0), None  # beams only represents what needs to be processed by the model in the next step
    cur_log_probs = torch.zeros((1,), dtype=torch.float32)  # (num of current beams,)
    cur_restricted_log_probs = cur_log_probs.clone()  # sum of restricted probabilities
    num_beams_over_time = []
    for n_cur in range(sample_len):
        logits, states = model.get_next_probs(beams, rnn_args=rnn_args, return_logits = True, device=device)
        next_log_probs = torch.log_softmax(logits, dim=-1)  # (num of current beams, vocab_size)
        next_log_probs[..., excluded_terms] = -float('inf')
        next_restricted_log_probs = torch.log_softmax(next_log_probs, dim=-1)
        next_log_probs = cur_log_probs.unsqueeze(-1) + next_log_probs
        next_log_probs = next_log_probs.view(-1)
        next_restricted_log_probs = cur_restricted_log_probs.unsqueeze(-1) + next_restricted_log_probs
        next_restricted_log_probs = next_restricted_log_probs.view(-1)

        if isinstance(num_beams, int):
            next_restricted_log_probs = top_k_top_p_filtering(next_restricted_log_probs, top_k=num_beams, is_log_prob=True)
        else:  # isinstance(num_beams, float)
            num_beams_cur = interp_func(num_beams, n_cur, sample_len)
            next_restricted_log_probs = top_k_top_p_filtering(next_restricted_log_probs, top_p=num_beams_cur, is_log_prob=True)

        next_log_probs = next_log_probs.masked_fill(next_restricted_log_probs == -float('inf'), -float('inf'))
        indices = torch.arange(0, next_log_probs.shape[0], device=beams.device)[next_log_probs != -float('inf')]
        seq_inds = torch.div(indices, vocab_size, rounding_mode='trunc')  # equivalent to: indices // args.vocab_size
        beams = (indices % vocab_size).unsqueeze(-1)
        cur_log_probs = next_log_probs[indices]
        cur_restricted_log_probs = next_restricted_log_probs[indices]
        rnn_args = states
        if isinstance(rnn_args, tuple):
            rnn_args = rnn_args[0][..., seq_inds, :], rnn_args[1][..., seq_inds, :]
        else:
            rnn_args = rnn_args[..., seq_inds, :]

        num_beams_over_time.append(cur_log_probs.shape[0])

    logits, states = model.get_next_probs(beams, rnn_args=rnn_args, device=device)
    next_log_probs = cur_log_probs.unsqueeze(-1) + torch.log_softmax(logits, dim=-1)
    return {
        "dist_lower_bound": next_log_probs.exp().sum(dim=0),
        "true_coverage": cur_log_probs.exp().sum(),
        "restricted_coverage": cur_restricted_log_probs.exp().sum(),
        "num_beams": num_beams_over_time,
    }


#######################################################################
# Sampling orchestration function
#######################################################################


@torch.no_grad()
def sample(
    args,
    dataloader,
    model = None,
    **kwargs,
):
    """Sample from any of these methods given an
    input dataloader, arguments, and potentially a model

    :dataloader: TODO
    :args: TODO
    :model: TODO
    :: TODO
    :returns: TODO

    """
    args.model = model; print();
    output = {"sample_estimates":[]}

    def _tensor_output(key, data,output=output):
        if key not in output: output[key] = []
        output[key].append(torch.Tensor([db[key] for db in data]))

    def _stack_output(key, data,output=output):
        if key not in output: output[key] = []
        output[key].append(torch.stack([
            torch.Tensor(db[key]) for db in data]))

    def _consolidate_output(key,output=output):
        output[key] = torch.cat(output[key])

    for dbatch in tqdm(dataloader, disable=args.disable_tqdm):
        data_list = []
        data_batch =[dbatch[i,:args.hist_len] for i in range(dbatch.shape[0])]

        for i in range(dbatch.shape[0]):
            if i%10 == 0 and args.disable_tqdm:
                print(".",end="",flush=True)
            sample = data_batch[i]
            args.sample_len = args.total_seq_len - args.hist_len
            kwargs = vars(args)
            # print(''.join([args.text_dict['id_to_char'][s] for s in sample.tolist()]))
            data_list.append(args.estimate_type(sample,**kwargs))


        if "beam_search" in args.estimate_type.__name__:
            _stack_output('num_beams',data_list)
            _stack_output('dist_lower_bound',data_list)
            _tensor_output('true_coverage',data_list)
            _tensor_output('restricted_coverage',data_list)
        else:
            output['sample_estimates'] += data_list


    if "beam_search" in args.estimate_type.__name__:
        _consolidate_output("num_beams")
        _consolidate_output("true_coverage")
        _consolidate_output("restricted_coverage")
        _consolidate_output("dist_lower_bound")
    else:
        _consolidate_output("sample_estimates")

    args.model = None
    output['metadata'] = vars(args)
    return output


