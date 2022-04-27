
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
import glob

sys.path.insert(1, '/home/showalte/research/prob_seq_queries/')

import numpy as np
import torch
from collections import defaultdict

# from experiments.train.shakespeare import main as shakespeare_main
# from experiments.train.stacks import main as stacks_main

from seq_queries.sample import sample
from seq_queries.model import get_model
from seq_queries.data import load_text_data, process_text_data, load_app_data, process_app_data, load_amazon_data, process_amazon_data
from seq_queries.arguments import get_args
from seq_queries.train import load_checkpoint, eval_epoch
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal
from seq_queries.experiments import sample_token_centric, sample_dynamic_target_token
#################################################################################
#   Function-Class Declaration
#################################################################################

if __name__ == "__main__":

    args = get_args(manual_config="scripts/model_performance.yaml")

    # # Amazon
    # text_dict= load_amazon_data(args.data_path)
    # args.text_dict = text_dict
    # print(text_dict['char_to_id'],flush=True)
    # train_dl, val_dl, test_dl = process_amazon_data(text_dict, args)

    # Apps
    text_dict= load_app_data(args.data_path)
    args.text_dict = text_dict
    print(text_dict['char_to_id'],flush=True)
    print("====="*10,flush=True)
    train_dl, val_dl, test_dl = process_app_data(text_dict, args)

    model = get_model(args)
    valid_perfs = []
    if args.checkpoint_path:
        checkpoints = glob.glob(f"{args.checkpoint_path}/*.pt")
        for checkpoint in checkpoints:
            load_checkpoint(args, model, checkpoint)
            new_valid = eval_epoch(args, model, val_dl, 0)
            print(new_valid)
            print("====="*10,flush=True)

    # args.proposal_func = lm_proposal
    # for exp in sample_experiments:
    #     args.hist_len = exp
    #     print("Hist length {} | Total Seq Length {} | Num samples: {} | Sample type: importance".format(args.hist_len,args.total_seq_len, args.num_mc_samples))
    #     estimates = sample_dynamic_target_token(args, val_dl, model)
    #     os.makedirs(f"data/importance_sampling/shakespeare/",exist_ok=True)
    #     write_pkl(estimates,f"data/importance_sampling/shakespeare/val-dl_importance-sampling_{args.hist_len}h_{args.total_seq_len}s_exc-dynamic.pkl")
    #     print("====="*10)


    # for exp in lb_experiments:
    #     args.hist_len, args.num_beams = exp
    #     print("Hist length {} | Total Seq Length {} | Coverage: {}".format(args.hist_len,args.total_seq_len, args.num_beams))
    #     estimates = sample_dynamic_target_token(args, val_dl, model)
    #     os.makedirs(f"data/beam_search/shakespeare/",exist_ok=True)
    #     write_pkl(estimates,f"data/beam_search/shakespeare/val-dl_beam-search_{args.hist_len}h_{args.total_seq_len}s_{args.num_beams}c_exc-dynamic.pkl")
    #     estimates = None
    #     print("====="*10)

    # output = sample_token_centric(args, val_dl, model)
    # output = sample(args,val_dl, model)
    # print(output['seqs'][0].shape)
    # print([''.join([text_dict['id_to_char'][c]
    #                 for c in output['seqs'][0][i,:].tolist()])
    #                for i in range(min(5,output['seqs'][0].shape[0]))]
    #         )
    # sys.exit(1)
    # estimates_or_lbs = evaluate_samples(args, model, output)
    # plot_estimates = [est_lbs[0].item() for est_lbs in estimates_or_lbs]




#################################################################################
#   Main Method
#################################################################################



