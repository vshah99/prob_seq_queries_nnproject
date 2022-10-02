
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
import glob

ROOT =os.path.abspath(os.path.join(__file__,"../../"))
sys.path.insert(1,ROOT)

import numpy as np
import torch
from collections import defaultdict

from seq_queries.sample import sample
from seq_queries.model import get_model
from seq_queries.arguments import get_args
from seq_queries.train import load_checkpoint, eval_epoch
from seq_queries.utils import write_pkl
from seq_queries.sample import lm_proposal
from seq_queries.experiments import sample_dynamic_target_token, prep_experiment
#################################################################################
#   Function-Class Declaration
#################################################################################

if __name__ == "__main__":

    args = get_args(manual_config="config/model_performance.yaml")

    dataset_name = "moocs"
    device = 7
    prep_dict = prep_experiment("config/model_performance.yaml",
                                dataset_name,
                                device=device)
    args = prep_dict['args']
    val_dl = prep_dict['val_dl']
    model = prep_dict['model']

    valid_perfs = []
    if args.checkpoint_path:
        checkpoints = glob.glob(f"{args.checkpoint_path}/*.pt")
        for checkpoint in checkpoints:
            epoch = load_checkpoint(args, model, checkpoint)
            new_valid = eval_epoch(args, model, val_dl, epoch)
            print(new_valid)
            print("====="*10,flush=True)

