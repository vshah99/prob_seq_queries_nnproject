#################################################################################
#
#             Project Title:  Flashy queries
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
from seq_queries.utils import write_pkl, write_json
from seq_queries.sample import lm_proposal, uniform_proposal, beam_search_lower_bound, mc_estimate, mc_pseudo_gt
from seq_queries.experiments import sample_dynamic_target_token, prep_experiment, flashy_query

#################################################################################
#   Function-Class Declaration
#################################################################################

device=6
sub_estimates = [10,100,1000]
folders = ["flashy"]
datasets = ['flashy_gpt2'] #'shakespeare'
config_path = "config/testing/sample.yaml"

for dataset_name in datasets:
    print("====="*10)
    print(f"* Running for dataset {dataset_name}")
    print("====="*10)
    extra_args = {}
    prep_dict = prep_experiment(config_path,
                                dataset_name,
                                device=device,
                                extra_args=extra_args)
    args = prep_dict['args']
    val_dl = prep_dict['val_dl']
    model = prep_dict['model']
    text_dict = args.text_dict
    args.text_dict = None
    print_args(vars(args))
    args.text_dict = text_dict
    print("====="*10)

    for folder in folders:
        args = copy.deepcopy(prep_dict['args'])
        args.estimate_type = mc_estimate
        args.use_gpt2 = True
        args.flashy=True
        args.proposal_func = lm_proposal
        args.sub_estimates = sub_estimates
        args.num_mc_samples = args.sub_estimates[-1]

        print("[{}] | Dataset: {} | Sample type: {} | Num samples: {} | Hist length {} | Total Seq Length {} "\
                .format(datetime.now(),dataset_name,folder,args.num_mc_samples,args.hist_len,args.total_seq_len))
        estimates = flashy_query(args, val_dl, model)
        os.makedirs(f"data/{folder}/{dataset_name}/val_dl/",exist_ok=True)
        estimates['metadata']['text_dict']['text'] = None
        args.sub_estimates = None
        args.num_mc_samples = sub_estimates[-1]

        # for e,d in estimates.items():
        #     if isinstance(d, (torch.Tensor, torch.LongTensor)):
        #         print(e, d.shape)
        # sys.exit(1)

        write_pkl(estimates,
        f"data/{folder}/{dataset_name}/val_dl/val-dl_{dataset_name}_{folder.replace('_','-')}_" +
        f"{args.num_mc_samples}mc.pkl")
        estimates=None
        print("====="*10)
