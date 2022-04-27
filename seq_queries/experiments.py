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

# @torch.no_grad()
# def entropy_vs_variance_bad_idea(
#     data_dict,
#     temperatures=[0.001,0.01,0.1,0.25,0.5,0.75,1,2,3,4,5,6,7,8,9,10,50,100],
#  ):
#     logits = data_dict['logits']

#     ref_probs = torch.gather(data_dict['sample_estimates'],-1,
#                     data_dict['excluded_terms'].repeat(
#                     (data_dict['sample_estimates'].shape[1],1)).T.unsqueeze(-1)).squeeze()
#     samples = defaultdict(dict)
#     for t in temperatures:
#         temp_logits = logits/t # (seqs x samples x steps x vocab)
#         # samples: (seqs, samples, steps)
#         temp_log_probs_by_step = torch.gather(F.log_softmax(temp_logits,dim=-1),-1,
#                                               data_dict['samples'].unsqueeze(-1)).squeeze()
#         # (seqs x samples)
#         temp_log_probs = temp_log_probs_by_step.sum(dim=-1)

#         entropy_est = -((torch.exp(temp_log_probs)/ref_probs)*temp_log_probs).mean(dim=-1)
#         # (seqs, 1)
#         variance_est = torch.var(torch.exp(temp_log_probs), dim=-1)
#         assert variance_est.shape == entropy_est.shape,\
#             f"Variance shape: {variance_est.shape} | Entropy shape: {entropy_est.shape}"
#         samples[t] = {"entropy":entropy_est, "variance": variance_est}
#     return samples

@torch.no_grad()
def sample_dynamic_target_token(
    args,
    dataloader,
    model = None,
    sample_artifacts=["sample_estimates",'q_log_prob','sample_est_var','sample_est_mean'],
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
            args.sample_len = args.total_seq_len - args.hist_len
            args.excluded_terms = [dbatch[i,args.total_seq_len].cpu().item()]
            kwargs = vars(args)
            # print(''.join([args.text_dict['id_to_char'][s] for s in sample.tolist()]))
            # bs_tree = BeamSearchSampleTree(args.text_dict)
            # args.bs_tree = bs_tree
            sample_output =args.estimate_type(sample,**kwargs)
            data_list.append(sample_output)

            # bs_tree = sample_output['tree']
            # print(bs_tree.depth_sizes)
            # print(bs_tree.depth_dict)
            # print()
            # bs_tree.prune()
            # print(bs_tree.depth_dict)
            # sys.exit(1)

        print("",flush=True)
        if "beam_search" in args.estimate_type.__name__:
            _add_output('num_beams',data_list)
            _add_output('dist_lower_bound',data_list)
            _tensor_output('true_coverage',data_list)
            _tensor_output('restricted_coverage',data_list)
        else:
            for add_out in sample_artifacts:
                _add_output(add_out,data_list)

    if "beam_search" in args.estimate_type.__name__:
        for c in search_artifacts: _consolidate_output(c)
    else:
        for c in sample_artifacts: _consolidate_output(c)

    args.model = None
    output['metadata'] = vars(args)
    output['excluded_terms'] = torch.cat(all_excluded_terms,dim=0).cpu()
    return output



def sample_token_centric(
    args,
    dataloader,
    model = None,
    **kwargs,
):
    """Samples queries where we know a certain token occurs within
    a certain range of the token in question. This process ensures that
    our estimates rest as high above zero as possible for long queries.

    :args: TODO
    :model: TODO
    :: TODO
    :returns: TODO

    """
    roster = {
        "beam_search": sample_beam_search,
        "mc_random": (mc_sample_random_batch if
        args.sample_args['coverage_type'] == "fixed_width"
                      else mc_sample_random_list),
        "mc_importance": mc_sample_importance,
    };
    sampler = roster[args.sample_type]
    args.model = model;
    output = {"settings":vars(args)}

    all_seqs = []; all_probs = []; all_beams = []; all_covs = []; all_sample_probs = []
    assert len(args.excluded) == 1,"Must only have one excluded token"
    excluded_token = args.excluded[0]
    for dbatch in dataloader:
        batched = False
        args.old_hist_len = args.hist_len
        if isinstance(args.hist_len,list):
            args.old_hist_len = args.hist_len[0]

        data_batch = dbatch[torch.eq(dbatch,excluded_token).sum(dim=-1).bool()]
        token_occ = torch.argmax((data_batch == excluded_token).float(), dim = -1, keepdim = True).flatten()
        data_batch = data_batch[token_occ > 0]; token_occ = token_occ[token_occ > 0]

        args.hist_len = torch.clamp(token_occ - args.old_hist_len-1,min=1)
        data_batch = data_batch[args.hist_len > 5]
        # print(data_batch.shape)
        args.hist_len = args.hist_len[args.hist_len > 1]
        args.total_seq_lens = (args.hist_len + (args.total_seq_lens-args.old_hist_len)).tolist()
        args.hist_len = args.hist_len.tolist()
        data_batch =[data_batch[i,:args.hist_len[i]].cpu() for i in range(data_batch.shape[0])]
        kwargs = vars(args)

        seqs, probs, beams_covs = sampler(data_batch,**kwargs)
        sample_probs = None
        if isinstance(probs,tuple):
            probs, sample_probs = probs

        #Tensors
        seqs = [seq.numpy() for seq in seqs]
        probs = ([None] if probs is None
                 else ([prob.numpy() for prob in probs] if isinstance(seqs, list)
                 else ([probs.numpy()])))
        sample_probs = ([None] if sample_probs is None
                 else ([sample_prob.numpy() for sample_prob in sample_probs] if isinstance(seqs, list)
                 else ([sample_probs.numpy()])))
        all_seqs += seqs
        all_probs += probs
        all_sample_probs += sample_probs

        # Lists
        if beams_covs is not None:
            all_beams += beams_covs[0]
            all_covs += beams_covs[1]

    output['beams'] = all_beams
    output['probs'] = all_probs
    output['sample_probs'] = all_sample_probs
    output['seqs'] = all_seqs
    output['covs'] = all_covs
    args.model = None

    return output

#################################################################################
#   Main Method
#################################################################################




