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
from .tree import BeamSearchSampleTree
from .sample import *
from .data import *
from .train import load_checkpoint, get_model

#################################################################################
#   Function-Class Declaration
#################################################################################


def prep_experiment(
    config_path,
    name="shakespeare",
    need_train=False,
    need_test=False,
    device=0,
 ):
    args = get_args(manual_config=config_path)
    name = name.lower()
    config_roster = {
        "amazon": {
            "checkpoint_path": "/home/showalte/research/prob_seq_queries/models/amazon/",
            "data_path": "/srv/disk00/samshow/amazon/amazon_text_dict.pkl",
            "hidden_size": 512,
            "seq_len": 15,
            "vocab_size": 30,
            "val_data_pct": 0.0001,
        },
        "apps": {
            "checkpoint_path": "/home/showalte/research/prob_seq_queries/models/apps/",
            "data_path": "data/apps/lsapp.tsv",
            "hidden_size": 512,
            "seq_len": 15,
            "vocab_size": 88,
            "val_data_pct": 0.001,
        },
        "shakespeare": {
            "checkpoint_path": "/home/showalte/research/prob_seq_queries/models/shakespeare/",
            "data_path": "data/shakespeare/shakespeare_input.txt",
            "hidden_size": 128,
            "seq_len": 100,
            "vocab_size": 68,
            "val_data_pct": 0.05,
        },
    }

    load_roster = {
        "amazon": load_amazon_data,
        "apps": load_app_data,
        "shakespeare": load_text_data,
    }
    assert name in load_roster,\
        "Dataset {} not found in roster"
    process_roster = {
        "amazon": process_amazon_data,
        "apps": process_app_data,
        "shakespeare": process_text_data,
    }
    for argument,details in config_roster[name].items():
        args.__dict__[argument] = details
    args.device=device
    text_dict= load_roster[name](args.data_path)
    args.text_dict = text_dict
    # print(text_dict['char_to_id'],flush=True)
    # print("====="*10)
    train_dl, val_dl, test_dl = process_roster[name](text_dict, args)

    model = get_model(args)
    if args.checkpoint_path:
        load_checkpoint(args, model)
    model.eval()
    print("====="*10)

    return {
        "train_dl": train_dl if need_train else None,
        "test_dl": test_dl if need_test else None,
        "val_dl": val_dl,
        "args": args,
        "model":model,
    }

@torch.no_grad()
def sample_dynamic_target_token(
    args,
    dataloader,
    model = None,
    sample_artifacts=["sample_estimates",'sample_est_var','sample_est_mean','model_iters'],
    hybrid_artifacts=["bs_lower_bound",'is_estimate','hybrid_bs_is_estimate','model_iters',
                      'hybrid_var','hybrid_mean','num_beams',],
    search_artifacts=['true_coverage','restricted_coverage','num_beams','dist_lower_bound'],
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
    artifact_store_roster = {
        "beam_search_is_hybrid": hybrid_artifacts,
        "beam_search_lower_bound":search_artifacts,
        "mc_estimate":sample_artifacts,
    }

    def _tensor_output(key, data,output=output):
        if key not in output: output[key] = []
        output[key].append(torch.Tensor([db[key] for db in data]))

    def _add_output(key, data,output=output):
        if key not in output: output[key] = []
        output_data =[db[key] for db in data]
        if not isinstance(output_data[0], (torch.Tensor, torch.LongTensor)):
            output_data =[torch.Tensor(db[key]) for db in data]
        output[key] += output_data

    def _consolidate_output(key,output=output):
        if isinstance(output[key],(torch.Tensor, torch.LongTensor)):
            return
        elif len(output[key][0].shape) == 1:
            output[key] = torch.stack(output[key]).squeeze()
        elif len(output[key][0].shape) >= 1:
            output[key] = torch.cat(output[key])

    all_excluded_terms = []
    artifacts = artifact_store_roster[args.estimate_type.__name__]
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
            # sample_output = variance_ablation(sample, **kwargs)
            # sys.exit(1)
            sample_output =args.estimate_type(sample,**kwargs)
            data_list.append(sample_output)


        print("",flush=True)
        assert args.estimate_type.__name__ in artifact_store_roster,\
            f"Estimate type {args.estimate_type.__name__} not found"
        artifacts = artifact_store_roster[args.estimate_type.__name__]
        for art in artifacts:
            _add_output(art,data_list)
        break

    for art in artifacts:
        _consolidate_output(art)
    # if "is_hybrid" in args.estimate_type.__name__:
    #     _consolidate_output('bs_lower_bound',stack=True)
    #     _consolidate_output('restricted_coverage',stack=False)
    #     _consolidate_output('true_coverage',stack=False)
    #     _consolidate_output('num_beams',stack=True)
    #     for c in hybrid_artifacts: _consolidate_output(c)
    # elif "beam_search" in args.estimate_type.__name__:
    #     _consolidate_output('dist_lower_bound',stack=False)
    #     _consolidate_output('restricted_coverage',stack=False)
    #     _consolidate_output('true_coverage',stack=False)
    #     _consolidate_output('num_beams',stack=True)
    # else:
    #     for c in sample_artifacts: _consolidate_output(c)

    args.model = None
    output['metadata'] = vars(args)
    output['excluded_terms'] = torch.cat(all_excluded_terms,dim=0).cpu()
    return output


#######################################################################
# Comparison of Variance
#######################################################################

def variance_ablation(
    hist, seq_len, model, interp_func,excluded_terms,
    batch_size, device, vocab_size, num_intervals=100, **kwargs
 ):
     # (beams)
     q_log_probs, p_log_cond = _get_joint_log_prob_of_all_seqs(
        hist, seq_len, model, interp_func,excluded_terms,
        batch_size, device, vocab_size, **kwargs)

     q_log_probs, inds = torch.sort(q_log_probs,
                                descending=True)
     p_log_cond = p_log_cond[inds]
     p_log_joint = (q_log_probs + p_log_cond)

     beam_search_step_size = int(q_log_probs.shape[0]/num_intervals)
     variances = [];

     global_var = None
     for i in range(num_intervals):
         part_p_joint = p_log_joint[i:].exp()
         part_q_probs = q_log_probs[i:].exp()
         part_q_probs /= part_q_probs.sum()

         w_x = (part_p_joint/part_q_probs)
         variance = (part_q_probs*torch.pow(w_x - w_x.mean(),2)).sum()
         if i == 0:
             global_var = variance
         variances.append(variance/global_var)

     return torch.Tensor(variances)


def _get_joint_log_prob_of_all_seqs(
    hist, seq_len, model, interp_func,excluded_terms,
    batch_size, device, vocab_size, **kwargs):
    """
    Examines the variance of the importance sampling estimate
    of the number of paths remaining before and after beam search.

    """

    assert len(excluded_terms) == 1,\
        "Ambiguous choice of excluded term to use"
    excluded_term = excluded_terms[0]
    all_seqs = torch.LongTensor(
        list(product(range(vocab_size),
            repeat=seq_len))
    )
    all_seqs = torch.cat(
        (all_seqs,
         torch.ones((all_seqs.shape[0],1))*excluded_term
        ), dim = 1
    ).long()

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
    model_log_probs = torch.gather(torch.log_softmax(logits[...,-(seq_len+1):,:],dim=-1),
                                   dim=-1, index=all_seqs.unsqueeze(-1)).squeeze(-1)
    q_log_prob = model_log_probs[...,:-1].sum(dim=-1)
    p_log_cond = model_log_probs[...,-1]

    return q_log_prob, p_log_cond










