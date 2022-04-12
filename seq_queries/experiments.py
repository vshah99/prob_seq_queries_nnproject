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
import torch
import torch.nn as nn

from tqdm import tqdm
from .data import load_text, process_data
from .model import CausalLM, MaskedLM
from .arguments import get_args
from .sample import *

#################################################################################
#   Function-Class Declaration
#################################################################################

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




