#################################################################################
#
#             Project Title:  Temperature ablations
#             Date:           2022-04-21
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

# from experiments.train.shakespeare import main as shakespeare_main
# from experiments.train.stacks import main as stacks_main

from seq_queries.sample import sample
from seq_queries.model import get_model
from seq_queries.data import load_amazon_data, process_amazon_data, load_app_data, process_app_data
from seq_queries.arguments import get_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal, uniform_proposal, beam_search_lower_bound, mc_estimate
from seq_queries.experiments import sample_dynamic_target_token
#################################################################################
#   Function-Class Declaration
#################################################################################

if __name__ == "__main__":

    # dataset_name = "amazon"
    dataset_name = "apps"
    args = get_args(manual_config="scripts/search_sample_baselines.yaml")

    # Amazon

    # # Mobile app data
    # text_dict= load_app_data(args.data_path)
    # args.text_dict = text_dict
    # print(text_dict['char_to_id'],flush=True)
    # print("====="*10,flush=True)
    # train_dl, val_dl, test_dl = process_app_data(text_dict, args)

    model = get_model(args)
    if args.checkpoint_path:
        load_checkpoint(args, model)
    model.eval()

    hist_lens = [12,13]

    args.estimate_type = mc_estimate
    args.proposal_func = lm_proposal
    for hist_len in hist_lens:
        args.hist_len = hist_len
        print("Hist length {} | Total Seq Length {} | Num samples: {} | |Sample type: importance".format(args.hist_len,args.total_seq_len, args.num_mc_samples))
        estimates = sample_dynamic_target_token(args, val_dl, model)
        os.makedirs(f"data/importance_sampling/{dataset_name}/val_dl/",exist_ok=True)
        write_pkl(estimates,f"data/importance_sampling/{dataset_name}/val_dl/val-dl_importance-sampling_{args.hist_len}h_{args.total_seq_len}s_{args.num_mc_samples}samples_exc-dynamic.pkl")
        print("====="*10)

    args.proposal_func = lm_proposal
    for hist_len in hist_lens:
        args.hist_len = hist_len
        print("Hist length {} | Total Seq Length {} | Num samples: {} | |Sample type: importance".format(args.hist_len,args.total_seq_len, args.num_mc_samples))
        estimates = sample_dynamic_target_token(args, val_dl, model)
        os.makedirs(f"data/random_sampling/{dataset_name}/val_dl/",exist_ok=True)
        write_pkl(estimates,f"data/random_sampling/{dataset_name}/val_dl/val-dl_random-sampling_{args.hist_len}h_{args.total_seq_len}s_{args.num_mc_samples}samples_exc-dynamic.pkl")
        print("====="*10)


    args.estimate_type = beam_search_lower_bound
    args.num_beams = 1.0
    for hist_len in hist_lens:
        args.hist_len = hist_len
        print("Hist length {} | Total Seq Length {} | Num samples: {} | |Sample type: importance".format(args.hist_len,args.total_seq_len, args.num_mc_samples))
        estimates = sample_dynamic_target_token(args, val_dl, model)
        os.makedirs(f"data/ground_truth/{dataset_name}/val_dl/",exist_ok=True)
        write_pkl(estimates,f"data/ground_truth/{dataset_name}/val_dl/val-dl_ground-truth_{args.hist_len}h_{args.total_seq_len}s_exc-dynamic.pkl")
        print("====="*10)

        # sequence length
        # vocabulary size
        # total sequence length
        # hist length




