#################################################################################
#
#             Project Title:  Sequential Queries
#             Author:         Sam Showalter
#             Date:           2022.04.11
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import numpy as np
from collections import defaultdict
from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
# from .data import load_text, process_data
from .model import CausalLM, MaskedLM
from .arguments import get_args
from .sample import *
from .tree import BeamSearchSampleTree

#################################################################################
#   Function-Class Declaration
#################################################################################

@torch.no_grad()
def sample_dynamic_target_token(
    args,
    dataloader,
    model = None,
    sample_artifacts=["sample_estimates",'q_log_prob','sample_est_var','sample_est_mean'],
    hybrid_artifacts=["bs_lower_bound",'is_estimate','hybrid_bs_is_estimate','model_runs',
                      'hybrid_var','hybrid_mean'],
    search_artifacts = ["num_beams","true_coverage","restricted_coverage","dist_lower_bound",],
    **kwargs,):
    """Sample from any of these methods given an
    input dataloader, arguments, and potentially a model

    :dataloader: TODO
    :args: TODO
    :model: TODO
    :: TODO
    :returns: TODO

    """
    args.model = model; print();
    output = {}

    def _tensor_output(key, data,output=output):
        if key not in output: output[key] = []
        output[key].append(torch.Tensor([db[key] for db in data]))

    def _add_output(key, data,output=output):
        if key not in output: output[key] = []
        output_data =[db[key] for db in data]
        if not isinstance(output_data[0], (torch.Tensor, torch.LongTensor)):
            output_data =[torch.Tensor(db[key]) for db in data]
        output[key] += output_data

    def _consolidate_output(key,output=output, stack=True):
        if stack:
            output[key] = torch.stack(output[key])
        else:
            output[key] = torch.cat(output[key])

    all_excluded_terms = []
    for dbatch in tqdm(dataloader, disable=args.disable_tqdm):
        data_list = []
        var_list = []
        mean_list = []
        all_excluded_terms.append(dbatch[:,args.total_seq_len].cpu())
        data_batch =[dbatch[i,:args.hist_len] for i in range(dbatch.shape[0])]

        for i in range(dbatch.shape[0]):
            # print(i)
            if i%10 == 0 and args.disable_tqdm:
                print(".",end="",flush=True)
            sample = data_batch[i]
            args.seq_len = args.total_seq_len - args.hist_len
            args.excluded_terms = [dbatch[i,args.total_seq_len].cpu().item()]
            kwargs = vars(args)
            # print(''.join([args.text_dict['id_to_char'][s] for s in sample.tolist()]))
            # bs_tree = BeamSearchSampleTree(args.text_dict)
            # args.bs_tree = bs_tree
            sample_output =args.estimate_type(sample,**kwargs)
            data_list.append(sample_output)


        print("",flush=True)
        if "is_hybrid" in args.estimate_type.__name__:
            for add_out in hybrid_artifacts:
                _add_output(add_out,data_list)
        elif "beam_search" in args.estimate_type.__name__:
            _add_output('num_beams',data_list)
            _add_output('dist_lower_bound',data_list)
            _tensor_output('true_coverage',data_list)
            _tensor_output('restricted_coverage',data_list)
        else:
            for add_out in sample_artifacts:
                _add_output(add_out,data_list)

    if "is_hybrid" in args.estimate_type.__name__:
        for c in hybrid_artifacts: _consolidate_output(c)
    elif "beam_search" in args.estimate_type.__name__:
        for c in search_artifacts: _consolidate_output(c)
    else:
        for c in sample_artifacts: _consolidate_output(c)

    args.model = None
    output['metadata'] = vars(args)
    output['excluded_terms'] = torch.cat(all_excluded_terms,dim=0).cpu()
    return output


#######################################################################
# Comparison of Variance
#######################################################################

def variance_ablation(
    hist, seq_len, model, interp_func,excluded_terms,
    batch_size, device, vocab_size, num_intervals=1000, **kwargs
 ):
     # (beams)
     all_log_probs = _get_joint_log_prob_of_all_seqs(
        hist, seq_len, model, interp_func,excluded_terms,
        batch_size, device, vocab_size, **kwargs)
     all_log_probs, inds = torch.sort(all_log_probs,
                                      descending=True)

     beam_search_step_size = int(all_log_probs.shape[0]/num_intervals)
     variances = []

     for i in range(num_intervals):
         q_log_prob = all_log_probs[i*beam_search_step_size:]
         q_prob = q_log_prob.exp()
         q_prob = q_prob/q_prob.sum()
         variances.append(q_prob.var())

     return torch.Tensor(variances)


def _get_joint_log_prob_of_all_seqs(
    hist, seq_len, model, interp_func,excluded_terms,
    batch_size, device, vocab_size, **kwargs):
    """
    Examines the variance of the importance sampling estimate
    of the number of paths remaining before and after beam search.

    """

    all_seqs = torch.LongTensor(
        list(product(range(vocab_size),
                    repeat=seq_len))
    )

    hists=hist.unsqueeze(0).expand(all_seqs.shape[0], -1)
    logits, hidden_state = model.get_next_probs(
        torch.cat((hists,all_seqs), dim=-1),
        return_forward_only=True,
        rnn_args=None,
        device=device,
        max_batch_size=batch_size,
        return_logits=True,
    )

    # All but the last probability q(x_1:k)
    model_log_prob = torch.log_softmax(logits, dim=-1)[..., -(seq_len):, :]
    model_log_prob = torch.gather(model_log_prob, dim=-1,
                                  index=all_seqs.unsqueeze(-1)).squeeze(-1).sum(dim=-1)

    return model_log_prob










