#################################################################################
#
#             Project Title:  Scratch Work
#             Author:         Sam Showalter
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy

import numpy as np
import torch

# from experiments.train.shakespeare import main as shakespeare_main
# from experiments.train.stacks import main as stacks_main

from seq_queries.sample import sample
from seq_queries.model import get_model
from seq_queries.data import load_text, process_data
from seq_queries.arguments import get_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.experiments import sample_token_centric, sample_dynamic_target_token
#################################################################################
#   Function-Class Declaration
#################################################################################


if __name__ == "__main__":

    sample_type = "beam_search"
    args = get_args(manual_config="config/testing/sample.yaml")
    text_dict= load_text(args.data_path)
    args.text_dict = text_dict
    print(text_dict['char_to_id'])
    train_dl, val_dl, test_dl = process_data(text_dict, args)
    model = get_model(args)
    if args.checkpoint_path:
        load_checkpoint(args, model)
    model.eval()

    #(hist_len, num_mc_samples=100000)
    sample_experiments = [25,24,23,22,21,20]


    # For all sequences, target is the 31st token
    # (hist_len,coverage, total_seq_len=30,estimate_type=search)
    lb_experiments = [
        (28,0.10),
        (25,0.99),
        (24,0.99),
        (23,0.99),
        (22,0.95),
        (21,0.90),
        (20,0.90),
    ]

    for exp in sample_experiments:
        args.hist_len = exp
        print("Hist length {} | Total Seq Length {} | Num samples: {} | Sample type: random".format(args.hist_len,args.total_seq_len, args.num_mc_samples))
        estimates = sample_dynamic_target_token(args, val_dl, model)
        os.makedirs(f"data/random_sampling/shakespeare/",exist_ok=True)
        write_pkl(estimates,f"data/random_sampling/shakespeare/val-dl_random-sampling_{args.hist_len}h_{args.total_seq_len}s_{args.num_beams}c_exc-dynamic.pkl")
        print("====="*10)



    # for exp in sample_experiments:
    #     args.hist_len = exp
    #     print("Hist length {} | Total Seq Length {} | Num samples: {} | Sample type: importance".format(args.hist_len,args.total_seq_len, args.num_mc_samples))
    #     estimates = sample_dynamic_target_token(args, val_dl, model)
    #     os.makedirs(f"data/importance_sampling/shakespeare/",exist_ok=True)
    #     write_pkl(estimates,f"data/importance_sampling/shakespeare/val-dl_importance-sampling_{args.hist_len}h_{args.total_seq_len}s_{args.num_beams}c_exc-dynamic.pkl")
    #     print("====="*10)


    # for exp in lb_experiments:
    #     args.hist_len, args.num_beams = exp
    #     print("Hist length {} | Total Seq Length {} | Coverage: {}".format(args.hist_len,args.total_seq_len, args.num_beams))
    #     estimates = sample_dynamic_target_token(args, val_dl, model)
    #     os.makedirs(f"data/beam_search/shakespeare/",exist_ok=True)
    #     write_pkl(estimates,f"data/beam_search/shakespeare/val-dl_beam-search_{args.hist_len}h_{args.total_seq_len}s_{args.num_beams}c_exc-dynamic.pkl")
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



