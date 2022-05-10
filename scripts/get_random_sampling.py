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

device=5
sub_estimates = [10,100,1000]
model_budget = True
folders = ["random_sampling"]
datasets = ['apps']#,'shakespeare',"amazon","apps"]
config_path = "config/testing/sample.yaml"
lengths = {
    "amazon":[(h,15) for h in reversed(range(5,14,1))],
    "apps":[(h,15) for h in reversed(range(5,14,1))],
    "shakespeare": [(h,20) for h in reversed(range(5,19,1))] + [(10,35),(10,60)],
}

for dataset_name in datasets:
    len_info = lengths[dataset_name]
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
    args.estimate_type = mc_estimate
    args.proposal_func = uniform_proposal
    args.sub_estimates = sub_estimates
    args.num_mc_samples = args.sub_estimates[-1]
    text_dict = args.text_dict
    args.text_dict = None
    print_args(vars(args))
    args.text_dict = text_dict
    print("====="*10)

    for folder in folders:
        for hist_len,total_seq_len in len_info:
            args.hist_len = hist_len
            args.total_seq_len = total_seq_len
            args.sub_estimates = [10,100,1000]
            args.num_mc_samples = args.sub_estimates[-1]

            if model_budget:
                args.model_budget_filepath = (f"/home/showalte/research/prob_seq_queries/" +
                                            f"data/beam_search_is_hybrid/{dataset_name}/val_dl/val-dl_" +
                    f"{dataset_name}_beam-search-is-hybrid_{args.hist_len}h_{args.total_seq_len}s_{args.num_mc_samples}mc.pkl")
                try:
                    assert os.path.exists(args.model_budget_filepath),\
                        f"Model budget filepath {args.model_budget_filepath} does not exist"
                except Exception as e:
                    print(args.model_budget_filepath)
                    print(e)
                    print("====="*10)
                    continue

            print("Dataset: {} | Sample type: {} | Num samples: {} | Hist length {} | Total Seq Length {}"\
                  .format(dataset_name,folder,args.num_mc_samples,args.hist_len,args.total_seq_len))
            estimates = sample_dynamic_target_token(args, val_dl, model)
            os.makedirs(f"data/{folder}/{dataset_name}/val_dl/",exist_ok=True)
            estimates['metadata']['text_dict']['text'] = None
            args.sub_estimates = sub_estimates
            args.num_mc_samples = sub_estimates[-1]

            # for e,d in estimates.items():
            #     if isinstance(d, (torch.Tensor, torch.LongTensor)):
            #         print(e, d.shape)
            # sys.exit(1)

            write_pkl(estimates,
                    f"data/{folder}/{dataset_name}/val_dl/val-dl_{dataset_name}_{folder.replace('_','-')}_" +
                    f"{args.hist_len}h_{args.total_seq_len}s_{args.num_mc_samples}mc{'_' + 'model-budget' if args.model_budget_filepath else  ''}.pkl")
            estimates=None
            print("====="*10)





#################################################################################
#   Main Method
#################################################################################



