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
from seq_queries.arguments import get_args, print_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal, uniform_proposal, beam_search_lower_bound, mc_estimate
from seq_queries.experiments import sample_dynamic_target_token, prep_experiment

#################################################################################
#   Function-Class Declaration
#################################################################################

device=0
folders = ["ground_truth"]
datasets = ['moocs']
config_path = "config/testing/sample.yaml"
lengths = {
    "moocs":[(13,15),(12,15)],
    "amazon":[(13,15),(12,15)],#,(11,15)],
    "apps": [(13,15),(12,15)],
    "shakespeare": [(18,20),(17,20)],
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
    args.estimate_type = beam_search_lower_bound
    args.min_variance = False
    args.num_beams = 0.0
    text_dict = args.text_dict
    args.text_dict = None
    print_args(vars(args))
    args.text_dict = text_dict
    print("====="*10)

    for folder in folders:
        for hist_len,total_seq_len in len_info:
            args.hist_len = hist_len
            args.total_seq_len = total_seq_len
            print("Dataset: {} | Sample type: {} | Hist length {} | Total Seq Length {}".format(dataset_name,folder,args.hist_len,args.total_seq_len))
            estimates = sample_dynamic_target_token(args, val_dl, model)
            os.makedirs(f"data/{folder}/{dataset_name}/val_dl/",exist_ok=True)
            estimates['metadata']['text_dict']['text'] = None
            write_pkl(estimates,
                    f"data/{folder}/{dataset_name}/val_dl/val-dl_{dataset_name}_{folder.replace('_','-')}_{args.hist_len}h_{args.total_seq_len}s.pkl")
            estimates=None
            print("====="*10)





#################################################################################
#   Main Method
#################################################################################



