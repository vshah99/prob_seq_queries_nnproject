import math
import sys
import copy
import yaml
import argparse
import torch
import pickle as pkl
import json
import ast

from datetime import datetime

from .utils import print_log, read_yaml
from .sample import (
    mc_estimate, beam_search_lower_bound,beam_search_is_hybrid, uniform_proposal, lm_proposal,
    mc_pseudo_gt, geom_interp, lin_interp)


#######################################################################
# Utilities for argparse
#######################################################################

def _str2estimate(estimate):
    assert estimate in ["search","sample","search_sample"],\
        "Estimate must be [search, sample, search_sample], got {}".format(estimate)
    roster = {"sample":mc_estimate,
              "search":beam_search_lower_bound,
              "search_sample":beam_search_is_hybrid,
              "sample_pseudo_gt":mc_pseudo_gt,
              }
    return roster[estimate]

def _str2interp_func(interp):
    assert interp in ["geometric","linear"],\
        "Interpolation function must be [geometric, linear], got {}".format(interp)
    roster = {"geometric":geom_interp,
              "linear":lin_interp,
              }
    return roster[interp]

def _str2proposal(proposal):
    assert proposal in ["uniform","lm"],\
        "Proposal must be [uniform, lm], got {}".format(proposal)
    roster = {"uniform":uniform_proposal,
              "lm":lm_proposal,
              }
    return roster[proposal]

def _str2bool(item):
    item =1 if item in ['true','True'] else 0
    return bool(item)

def _int_or_float(item):
    try: item = int(item)
    except:
        return float(item)
    return item

def _int_or_float_or_list(item):
    assert isinstance(item, (int,float,list)),\
        f"Item was not int, float or list: got {type(item)}"
    return item

def _str2list(str_list):
    return ast.literal_eval(str_list)

def _merge_configs(primary, secondary):
    """
    Super simple configuration update
    function. Not great, but good enough
    for what I am doing

    Ignore functions
    """
    for k,v in primary.items():

        if (k not in secondary):
            secondary[k] = v

        # Key for primary dictionary is also a dictionary
        # and key present in both dictionaries
        elif isinstance(v, dict) and isinstance(secondary[k], dict):
            secondary[k] = _merge_configs(primary[k], secondary[k])

        # Key is present in both but not nested
        # Only if current object is not a function
        # elif not hasattr(secondary[k], '__call__'):
        else:
            secondary[k] = v

    return secondary



def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    group.add_argument("--seed", type=int, default=1234321, help="Seed for all random processes.")
    group.add_argument("--dont_print_args", type=_str2bool,default=False, help="Specify to disable printing of arguments.")
    group.add_argument("--cuda", type=_str2bool,default=True, help="Convert model and data to GPU.")
    group.add_argument("--device_num", type=int, default=0, help="Should cuda be enabled, this is the GPU id to use.")
    group.add_argument("--data_path",  type=str, default="./data/imdb.txt", help="Path to training data file.")
    group.add_argument("--dataset_phase_shift_path",  type=str, default="/srv/disk00/samshow/amazon/amazon_phase_trans.pkl", help="Dataset phase shift path for user behavior data")
    group.add_argument("--min_phase_shift",  type=int, default=0, help="Minimum number of phase shifts needed in each sequence")
    group.add_argument("--train_data_pct", type=float, default=0.9, help="Percent of data used for training.")
    group.add_argument("--val_data_pct", type=float, default=0.05, help="Percent of data used for validation.")
    group.add_argument("--seq_len", type=int, default=100, help="Length of sequences of tokens to process.")
    group.add_argument("--do_test",type=_str2bool, default=True, help="Perform evaluation on testing set.")
    group.add_argument("--do_valid",type=_str2bool, default=True, help="Perform evaluation on valid set.")

def model_config_args(parser):
    group = parser.add_argument_group("Model configuration arguments.")
    group.add_argument("--rnn_type", type=str, default="LSTM", help="RNN type to use in model. Supported values are 'RNN', 'LSTM', and 'GRU'.")
    group.add_argument("--use_gpt2", type=_str2bool, default=False, help="If we are using pretrained GPT-2")
    group.add_argument("--masked_lm",type=_str2bool,default=True, help="If enabled, makes RNN bidirectional and turns it into a Masked Language Model rather than a Causal Language Model.")
    group.add_argument("--vocab_size", type=int, default=-1, help="Number of unique vocabulary terms to embed and predict. A value of -1 means this will be inferred by the dataset.")
    group.add_argument("--hidden_size", type=int, default=32, help="Size of hidden state of RNN.")
    group.add_argument("--num_layers", type=int, default=3, help="Number of RNN layers.")
    group.add_argument("--dropout", type=float, default=0.2, help="Dropout rate to be applied to all supported layers during training.")

def training_args(parser):
    group = parser.add_argument_group("Training specification arguments.")
    group.add_argument("--checkpoint_path", type=str, default=None, help="Path to folder that contains model checkpoints. Will take the most recent one.")
    group.add_argument("--finetune", type=_str2bool,default=True, help="Will load in a model from the checkpoint path to finetune.")
    group.add_argument("--train_epochs", type=int, default=40, help="Number of epochs to iterate over for training.")
    group.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers for data loaders.")
    group.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch.")
    group.add_argument("--log_interval", type=int, default=100, help="Number of batches to complete before printing intermediate results.")
    group.add_argument("--save_epochs", type=int, default=1, help="Number of training epochs to complete between model checkpoint saves.")
    group.add_argument("--val_metrics", type=_str2list, default=['accuracy'], help="Validation metrics to track for models")
    group.add_argument("--valid_epochs", type=int, default=1, help="Valid epochs for evaluating on validation dataset.")
    group.add_argument("--optimizer", type=str, default="adam", help="Type of optimization algorithm to use.")
    group.add_argument("--grad_clip", type=float, default=1.0, help="Threshold for gradient clipping.")
    group.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    group.add_argument("--weight_decay", type=float, default=0.0, help="L2 coefficient for weight decay.")
    group.add_argument("--warmup_pct", type=float, default=0.01, help="Percentage of 'train_iters' to be spent ramping learning rate up from 0.")
    group.add_argument("--lr_decay_style", type=str, default="constant", help="Decay style for the learning rate, after the warmup period. Choices: 'constant', 'monotonic', and 'cosine'.")
    group.add_argument("--dont_shuffle", type=_str2bool, help="Don't shuffle training and validation dataloaders.")

def evaluation_args(parser):
    group = parser.add_argument_group("Evaluation specification arguments.")
    #group.add_argument("--", type=, default=, help="")pin_test_memory

def sampling_args(parser):
    group = parser.add_argument_group("Sampling specification arguments.")
    group.add_argument("--hist_len", type=int, default=10, help="Length of conditioning context for sequence")
    group.add_argument("--total_seq_len", type=int, default=15, help="List[int] or int for total sequence lengths")
    group.add_argument("--estimate_type", type=_str2estimate,default="sample", help="Get estimate type")
    group.add_argument("--proposal_func",  type=_str2proposal,default="uniform", help="Get proposal distribution for sampling")
    group.add_argument("--interp_func",  type=_str2interp_func,default="linear", help="Get inpterpolation function for search coverage")
    group.add_argument("--excluded_terms", type=_str2list, default=[], help="List of excluded terms")
    group.add_argument("--sub_estimates", type=_str2list, default=[], help="Sub-estimates (to track noise), all must be <= num_samples")
    group.add_argument("--min_num_mc_samples", type=int, default=10000, help="Minimum number of samples for pseudo ground truth")
    group.add_argument("--max_num_mc_samples", type=int, default=100000, help="Maximum number of samples for pseudo ground truth")
    group.add_argument("--variance_epsilon", type=float, default=5e-6, help="Variance threshold to stop sampling for pseudo ground truth")
    group.add_argument("--model_budget_filepath", type=str, default=None, help="Filepath to extract model budgets for another run (usually hybrid file for imp. samp.)")
    group.add_argument("--store_intermediate_lbs", type=_str2bool,default=True,help="Store intermediate lower bounds.")
    group.add_argument("--max_num_queries", type=int, default=None, help="Maximum numbers of queries to take (for faster experiments)")
    group.add_argument("--top_k", type=int, default=1, help="Top k beams/samples to take")
    group.add_argument("--top_p", type=_int_or_float, default=1, help="Top p coverage to take")
    group.add_argument("--min_variance", type=_str2bool, default=False, help="Use minimum variance to set beam search beam widths")
    group.add_argument("--min_var_reduction", type=float, default=0.0, help="Minimum variance reduction for minimum variance technique (otherwise, take all beams)")
    group.add_argument("--hybrid_max_num_beams", type=int, default=1500, help="Maximum_number of beams the hybrid can hold at each step")
    group.add_argument("--num_beams", type=_int_or_float, default=10, help="Beam coverage (or number)")
    group.add_argument("--num_mc_samples", type=int, default=10, help="Number of MC samples")
    group.add_argument("--disable_tqdm", type=_str2bool,default=False,help="Disable tqdm monitoring runs for samplers")

def print_args(args):
    max_arg_len = max(len(k) for k, v in args.items())
    key_set = sorted([k for k in args.keys()])
    for k in key_set:
        v = args[k]
        print_log("{} {} {}".format(
            k,
            "." * (max_arg_len + 3 - len(k)),
            v,
        ))

def get_args(manual_config = None):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    general_args(parser)
    model_config_args(parser)
    training_args(parser)
    #evaluation_args(parser)
    sampling_args(parser)

    def yaml2argparse(astr):
        # custom convert_arg_line_to_args method
        # convert 'names: [v1,v2]' into ['--names', v1, '--names', v2, ...]
        alist = []
        if ':' not in astr:
            return astr
        field,value = astr.split(':')
        value = value.strip()
        field = '--'+field
        alist.extend([field, value])
        return alist

    parser.convert_arg_line_to_args = yaml2argparse
    if manual_config is not None:
        args, _ = parser.parse_known_args(['@' + manual_config])
    else:
        args = parser.parse_args()

    args.shuffle = not args.dont_shuffle
    if not args.dont_print_args:
        print_args(vars(args))
        print("====="*10)

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.device_num))
    else:
        args.device = torch.device("cpu")

    return args

