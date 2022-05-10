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
from seq_queries.data import load_amazon_data, process_amazon_data, load_app_data, process_app_mooc_data
from seq_queries.arguments import get_args, print_args
from seq_queries.train import load_checkpoint
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal, uniform_proposal, beam_search_lower_bound, mc_estimate, beam_search_is_hybrid
from seq_queries.experiments import sample_dynamic_target_token, prep_experiment

#################################################################################
#   Function-Class Declaration
#################################################################################

device=1
folders = ["beam_search"]
datasets = ['shakespeare','amazon','apps','moocs']
model_budget = False
config_path = "config/testing/sample.yaml"
lengths_coverage = {
    "moocs":[(11,15,0.9),(10,15,0.9),(9,15,0.9),(8,15,0.9)],
    "amazon":[(11,15,0.9),(10,15,0.9),(9,15,0.9),(8,15,0.9),(7,15,0.80)],
    "apps":[(13,15,0.98),(12,15,0.92), (11,15,0.9),(10,15,0.85)],
    "shakespeare":[(16,20,0.9),(15,20,0.9),(14,20,0.9),(13,20,0.85)],
}

for dataset_name in datasets:
    len_info = lengths_coverage[dataset_name]
    print("====="*10)
    print(f"* Running for dataset {dataset_name}")
    print("====="*10)
    prep_dict = prep_experiment(config_path,
                                dataset_name,
                                device=device)
    prep_dict['args'].text_dict['text'] = None
    args = prep_dict['args']
    val_dl = prep_dict['val_dl']
    model = prep_dict['model']
    text_dict = args.text_dict
    args.text_dict = None
    print_args(vars(args))
    args.text_dict = text_dict
    print("====="*10)

    for folder in folders:
        for hist_len,total_seq_len,coverage in len_info:
            args = copy.deepcopy(prep_dict['args'])
            args.num_mc_samples = 1000 # For reading from hybrid correctly
            args.estimate_type = beam_search_lower_bound
            args.proposal_func = lm_proposal
            args.store_intermediate_lbs=True
            args.min_variance = False
            args.hist_len = hist_len
            args.total_seq_len = total_seq_len
            args.num_beams = float(coverage)

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

            print("Dataset: {} | Sample type: {} | Num Beams: {} | Hist length {} | Total Seq Length {}"\
                  .format(dataset_name,folder,args.num_beams,args.hist_len,args.total_seq_len))
            estimates = sample_dynamic_target_token(args, val_dl, model)
            os.makedirs(f"data/{folder}/{dataset_name}/val_dl/",exist_ok=True)
            estimates['metadata']['text_dict']['text'] = None
            args.num_beams = float(coverage)

            # for e,d in estimates.items():
            #     if isinstance(d, (torch.Tensor, torch.LongTensor)):
            #         print(e, d.shape)
            # sys.exit(1)

            write_pkl(estimates,
                    f"data/{folder}/{dataset_name}/val_dl/val-dl_{dataset_name}_" +
                    f"{folder.replace('_','-')}_{args.hist_len}h_{args.total_seq_len}s" +
                    f"_{args.num_beams + 'b' if not model_budget else 'model-budget'}.pkl")
            print("====="*10)





#################################################################################
#   Main Method
#################################################################################

# for e,d in estimates.items():
#     if isinstance(d, (torch.Tensor, torch.LongTensor)):
#         print(e, d.shape)
# sys.exit(1)



