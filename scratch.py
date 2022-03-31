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
#################################################################################
#   Function-Class Declaration
#################################################################################


if __name__ == "__main__":

    args = get_args(manual_config="config/testing/sample.yaml")
    text_dict= load_text(args.data_path)
    train_dl, val_dl, test_dl = process_data(text_dict, args)
    model = get_model(args)
    if args.checkpoint_path:
        load_checkpoint(args, model)
    model.eval()

    output = sample(val_dl, args, model)

#################################################################################
#   Main Method
#################################################################################



