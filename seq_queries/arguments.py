import argparse
import json
import torch

from .utils import print_log, _merge_configs, read_yaml, _int_or_float_or_list


def general_args(parser):
    group = parser.add_argument_group("General set arguments for miscelaneous utilities.")
    #group.add_argument("--json_config_path", default=None, help="Path to json file containing arguments to be parsed.")
    group.add_argument("--config", type=read_yaml, default = None, help="Config file to seed argparse from yaml file. WILL OVERWRITE OTHER ARGUMENTS")
    group.add_argument("--seed", type=int, default=1234321, help="Seed for all random processes.")
    group.add_argument("--dont_print_args", action="store_true", help="Specify to disable printing of arguments.")
    group.add_argument("--cuda", action="store_true", help="Convert model and data to GPU.")
    group.add_argument("--device_num", type=int, default=0, help="Should cuda be enabled, this is the GPU id to use.")
    group.add_argument("--data_path",  type=str, default="./data/imdb.txt", help="Path to training data file.")
    group.add_argument("--train_data_pct", type=float, default=0.9, help="Percent of data used for training.")
    group.add_argument("--valid_data_pct", type=float, default=0.05, help="Percent of data used for validation.")
    group.add_argument("--seq_len", type=int, default=100, help="Length of sequences of tokens to process.")
    group.add_argument("--do_test", action="store_true", help="Perform evaluation on testing set.")

def model_config_args(parser):
    group = parser.add_argument_group("Model configuration arguments.")
    group.add_argument("--rnn_type", type=str, default="LSTM", help="RNN type to use in model. Supported values are 'RNN', 'LSTM', and 'GRU'.")
    group.add_argument("--masked_lm", action="store_true", help="If enabled, makes RNN bidirectional and turns it into a Masked Language Model rather than a Causal Language Model.")
    group.add_argument("--vocab_size", type=int, default=-1, help="Number of unique vocabulary terms to embed and predict. A value of -1 means this will be inferred by the dataset.")
    group.add_argument("--hidden_size", type=int, default=32, help="Size of hidden state of RNN.")
    group.add_argument("--num_layers", type=int, default=3, help="Number of RNN layers.")
    group.add_argument("--dropout", type=float, default=0.2, help="Dropout rate to be applied to all supported layers during training.")

def training_args(parser):
    group = parser.add_argument_group("Training specification arguments.")
    group.add_argument("--checkpoint_path", type=str, default=None, help="Path to folder that contains model checkpoints. Will take the most recent one.")
    group.add_argument("--finetune", action="store_true", help="Will load in a model from the checkpoint path to finetune.")
    group.add_argument("--train_epochs", type=int, default=40, help="Number of epochs to iterate over for training.")
    group.add_argument("--num_workers", type=int, default=0, help="Number of parallel workers for data loaders.")
    group.add_argument("--batch_size", type=int, default=32, help="Number of samples per batch.")
    group.add_argument("--log_interval", type=int, default=100, help="Number of batches to complete before printing intermediate results.")
    group.add_argument("--save_epochs", type=int, default=1, help="Number of training epochs to complete between model checkpoint saves.")
    group.add_argument("--optimizer", type=str, default="adam", help="Type of optimization algorithm to use.")
    group.add_argument("--grad_clip", type=float, default=1.0, help="Threshold for gradient clipping.")
    group.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    group.add_argument("--weight_decay", type=float, default=0.0, help="L2 coefficient for weight decay.")
    group.add_argument("--warmup_pct", type=float, default=0.01, help="Percentage of 'train_iters' to be spent ramping learning rate up from 0.")
    group.add_argument("--lr_decay_style", type=str, default="constant", help="Decay style for the learning rate, after the warmup period. Choices: 'constant', 'monotonic', and 'cosine'.")
    group.add_argument("--dont_shuffle", action="store_true", help="Don't shuffle training and validation dataloaders.")

def evaluation_args(parser):
    group = parser.add_argument_group("Evaluation specification arguments.")
    #group.add_argument("--", type=, default=, help="")pin_test_memory

def sampling_args(parser):
    group = parser.add_argument_group("Sampling specification arguments.")
    group.add_argument("--hist_len", type=_int_or_float_or_list, default=10, help="Length of conditioning context for sequence")
    group.add_argument("--total_seq_lens", type=_int_or_float_or_list, default=15, help="List[int] or int for total sequence lengths")
    group.add_argument("--beam_widths", type=_int_or_float_or_list, default=0.5, help="List[int,float], float, or int for total sequence lengths")
    group.add_argument("--num_seqs", type=int, default=50, help="Number of sequences per sample")
    group.add_argument("--sample_type", default="beam_search", help="beam_search, mc_random, or mc_importance")
    group.add_argument("--sample_occ_range", type=int, default=4, help="Distance a gt token can be from the evaluated token in question")
    group.add_argument("--sample_args", type=dict, default={"coverage_type":"fixed_width"}, help="")
    group.add_argument("--excluded", type=dict, default={"coverage_type":"fixed_width"}, help="")
    group.add_argument("--disable_tqdm", action="store_false",help="Disable tqdm monitoring runs for samplers")

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
    parser = argparse.ArgumentParser()

    general_args(parser)
    model_config_args(parser)
    training_args(parser)
    #evaluation_args(parser)
    sampling_args(parser)

    args = parser.parse_args()

    args.shuffle = not args.dont_shuffle

    if manual_config is not None:
        args.config = read_yaml(manual_config)
    if args.config is not None:
        args.__dict__ = _merge_configs(args.config, args.__dict__)

    if not args.dont_print_args:
        print_args(vars(args))
        print("====="*10)

    if args.cuda and torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.device_num))
    else:
        args.device = torch.device("cpu")

    return args
