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

sys.path.insert(1, '/home/showalte/research/prob_seq_queries/')

import numpy as np
import torch
from collections import defaultdict


from seq_queries.sample import sample
from seq_queries.model import get_model
from seq_queries.data import load_amazon_data, process_amazon_data, load_app_data, process_app_data
from seq_queries.arguments import get_args, print_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal, uniform_proposal, beam_search_lower_bound, mc_estimate, beam_search_is_hybrid
from seq_queries.experiments import sample_dynamic_target_token, prep_experiment

#################################################################################
#   Function-Class Declaration
#################################################################################

device=3
num_mc_samples = 1000
folders = ["beam_search_is_hybrid"]
datasets = ["apps","amazon","shakespeare"]
config_path = "config/testing/sample.yaml"
lengths = {
    "amazon":[(h,15) for h in reversed(range(8,14,1))],
    "apps":[(h,15) for h in reversed(range(10,14,1))],
    "shakespeare": [(h,20) for h in reversed(range(14,18,1))],
}

for dataset_name in datasets:
    len_info = lengths[dataset_name]
    print("====="*10)
    print(f"* Running for dataset {dataset_name}")
    print("====="*10)
    prep_dict = prep_experiment(config_path,
                                dataset_name,
                                device=device)
    args = prep_dict['args']
    val_dl = prep_dict['val_dl']
    model = prep_dict['model']
    args.num_mc_samples = num_mc_samples
    args.estimate_type = beam_search_is_hybrid
    args.proposal_func = lm_proposal
    args.min_variance = True
    args.min_var_reduction = 0.1
    args.num_beams = 0.0
    args.sub_estimates = [10,100,1000]
    text_dict = args.text_dict
    args.text_dict = None
    print_args(vars(args))
    args.text_dict = text_dict
    print("====="*10)

    for folder in folders:
        for hist_len,total_seq_len in len_info:
            args.hist_len = hist_len
            args.total_seq_len = total_seq_len
            print("Dataset: {} | Sample type: {} | Num samples: {} | Hist length {} | Total Seq Length {}"\
                  .format(dataset_name,folder,args.num_mc_samples,args.hist_len,args.total_seq_len))
            estimates = sample_dynamic_target_token(args, val_dl, model)
            os.makedirs(f"data/{folder}/{dataset_name}/val_dl/",exist_ok=True)
            estimates['metadata']['text_dict']['text'] = None
            write_pkl(estimates,
                    f"data/{folder}/{dataset_name}/val_dl/val-dl_{dataset_name}_{folder.replace('_','-')}_{args.hist_len}h_{args.total_seq_len}s_{args.num_mc_samples}mc.pkl")
            estimates=None
            print("====="*10)





#################################################################################
#   Main Method
#################################################################################



