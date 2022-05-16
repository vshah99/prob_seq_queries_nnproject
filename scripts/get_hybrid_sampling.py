#################################################################################
#
#             Project Title:  Ground Truth Experiments
#             Author:         Sam Showalter
#             Date:           2022-04-30
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
from datetime import datetime

sys.path.insert(1, '/home/showalte/research/prob_seq_queries/')

import numpy as np
import torch
from collections import defaultdict

from seq_queries.model import get_model
from seq_queries.arguments import get_args, print_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal, uniform_proposal, beam_search_lower_bound, mc_estimate, beam_search_is_hybrid
from seq_queries.experiments import sample_dynamic_target_token, prep_experiment

#################################################################################
#   Function-Class Declaration
#################################################################################

device=3
sub_estimates = [10,100,1000]
folders = ["beam_search_is_hybrid"]
datasets = ['shakespeare','moocs','amazon','apps']
# datasets = ['wikitext']
# datasets = ['shakespeare']
max_num_queries = 1000
config_path = "config/testing/sample.yaml"
lengths = {

    # Long hybrid
    "moocs":[(h,15) for h in [10,7,6]],
    "amazon":[(h,15) for h in [10,7,6]],
    "apps":[(h,15) for h in [10,7,6]],
    "shakespeare": [(h,20) for h in [15,13,11]],

    # # Short hybrid
    "wikitext":[(h,15) for h in reversed(range(11,14,1))],
    # "moocs":[(h,15) for h in reversed(range(12,14,1))],
    # "amazon":[(h,15) for h in reversed(range(12,14,1))],
    # "apps":[(h,15) for h in reversed(range(12,14,1))],
    # "shakespeare": [(h,20) for h in reversed(range(17,19,1))],
}

for dataset_name in datasets:
    len_info = lengths[dataset_name]
    print("====="*10)
    print(f"* Running for dataset {dataset_name}")
    print("====="*10)
    extra_args = {"max_num_queries":max_num_queries}
    prep_dict = prep_experiment(config_path,
                                dataset_name,
                                device=device,
                                extra_args=extra_args)
    prep_dict['args'].text_dict['text'] = None
    args = prep_dict['args']
    if dataset_name=='wikitext':args.use_gpt2=True
    val_dl = prep_dict['val_dl']
    model = prep_dict['model']
    text_dict = args.text_dict
    args.text_dict = None
    print_args(vars(args))
    args.text_dict = text_dict
    print("====="*10)

    for folder in folders:
        for hist_len,total_seq_len in len_info:
            args = copy.deepcopy(prep_dict['args'])
            args.estimate_type = beam_search_is_hybrid
            args.proposal_func = lm_proposal
            args.min_variance = True
            args.max_num_tree_beams = 1500
            args.min_var_reduction = 0.1
            args.num_beams = 0.0
            args.sub_estimates = sub_estimates
            args.num_mc_samples = args.sub_estimates[-1]
            args.hist_len = hist_len
            args.total_seq_len = total_seq_len

            print("[{}] | Dataset: {} | Sample type: {} | Num samples: {} | Hist length {} | Total Seq Length {} | Max num queries: {}"\
                  .format(datetime.now(), dataset_name,folder,args.num_mc_samples,args.hist_len,args.total_seq_len,max_num_queries))
            estimates = sample_dynamic_target_token(args, val_dl, model)
            os.makedirs(f"data/{folder}/{dataset_name}/val_dl/",exist_ok=True)
            estimates['metadata']['text_dict']['text'] = None

            # for e,d in estimates.items():
            #     if isinstance(d, (torch.Tensor, torch.LongTensor)):
            #         print(e, d.shape)
            # sys.exit(1)

            args.num_mc_samples = sub_estimates[-1]
            write_pkl(estimates,
                    f"data/{folder}/{dataset_name}/val_dl/val-dl_{dataset_name}_" +
                    f"{folder.replace('_','-')}_{args.hist_len}h_{args.total_seq_len}s_" +
                    f"{args.num_mc_samples}mc" +
                    f"{f'_{args.max_num_queries}q' if args.max_num_queries else ''}.pkl")
            estimates=None
            print("====="*10)





#################################################################################
#   Main Method
#################################################################################


# for e,d in estimates.items():
#     if isinstance(d, (torch.Tensor, torch.LongTensor)):
#         print(e, d.shape)
# sys.exit(1)

