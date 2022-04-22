#################################################################################
#
#             Project Title:  Temperature ablations
#             Author:         Sam Showalter
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
from seq_queries.data import load_text_data, process_text_data, load_app_data, process_app_data
from seq_queries.arguments import get_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal
from seq_queries.experiments import sample_token_centric, sample_dynamic_target_token
#################################################################################
#   Function-Class Declaration
#################################################################################

if __name__ == "__main__":

    args = get_args(manual_config="config/testing/sample.yaml")
    text_dict= load_text_data(args.data_path)
    # text_dict= load_app_data(args.data_path, seq_len=args.seq_len)
    args.text_dict = text_dict
    print(text_dict['char_to_id'])
    train_dl, val_dl, test_dl = process_text_data(text_dict, args)
    # train_dl, val_dl, test_dl = process_app_data(text_dict, args)
    model = get_model(args)
    if args.checkpoint_path:
        load_checkpoint(args, model)
    model.eval()
    # estimates = sample_dynamic_target_token(args, val_dl, model, variance_only=True)

    #(hist_len, num_mc_samples=100000)
    temperatures = [0.05]

    for temp in temperatures:
        model.temperature = temp
        print("Hist length {} | Total Seq Length {} | Num samples: {} | Temperature: {} |Sample type: importance".format(args.hist_len,args.total_seq_len, args.num_mc_samples,temp))
        estimates = sample_dynamic_target_token(args, val_dl, model, variance_only=True)
        os.makedirs(f"data/importance_sampling/shakespeare/temp_ablation/",exist_ok=True)
        write_pkl(estimates,f"data/importance_sampling/shakespeare/temp_ablation/val-dl_importance-sampling_{args.hist_len}h_{args.total_seq_len}s_{temp}t_exc-dynamic.pkl")
        print("====="*10)

