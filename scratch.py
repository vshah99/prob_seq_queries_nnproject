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

from seq_queries.sample import sample, evaluate_samples
from seq_queries.model import get_model
from seq_queries.data import load_text, process_data
from seq_queries.arguments import get_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.experiments import sample_token_centric, sample_dynamic_experiment
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
    estimate = sample_dynamic_experiment(args,val_dl, model)

    sample_type_roster_long = {"mc_random":"random_sampling",
                          "mc_importance":"importance_sampling",
                          "beam_search":"beam_search"}
    sample_type_roster_short = {"mc_random":"mc-rand",
                          "importance_sampling":"mc-imp",
                          "beam_search":"gt"}
    write_pkl(estimate,f"data/{sample_type_roster_long[args.sample_type]}/shakespeare/val-dl_{sample_type_roster_short[args.sample_type]}_{args.hist_len}h_{args.total_seq_lens}s_{args.beam_widths}b_exc-dynamic.pkl")

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



