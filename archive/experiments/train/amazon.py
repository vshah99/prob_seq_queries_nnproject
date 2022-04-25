#################################################################################
#
#             Project Title:  Train Amazon LM
#             Author:         Sam Showalter
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os

import torch
import torch.nn as nn

# os.chdir("../../")
import sys
print(os.getcwd())
sys.path.insert(1, '/home/showalte/research/prob_seq_queries/')

from seq_queries.model import get_model
from seq_queries.data import load_amazon_data, process_amazon_data
from seq_queries.arguments import get_args
from seq_queries.train import *
from seq_queries.optim import *
#################################################################################
#   Function-Class Declaration
#################################################################################

def get_optimizer(model, args):

    optimizer = OPTIMIZERS[args.optimizer](
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,  # learning rate, weight decay, etc.
    )

    return optimizer

def train(args,train_dataloader, valid_dataloader):
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, optimizer, lr_scheduler = setup_model_and_optim(args, len(train_dataloader))
    print(model)
    sys.exit(1)

    report_model_stats(model)

    if args.finetune:
        epoch = load_checkpoint(args, model)
    else:
        epoch = 0
    original_epoch = epoch

    print_log("Starting training.")
    results = {"valid": [], "train": [], "test": []}

    while epoch < args.train_epochs:
        results["train"].append(train_epoch(args, model, optimizer, lr_scheduler, train_dataloader, epoch+1))

        if args.do_valid and ((epoch+1) % args.valid_epochs == 0):
            new_valid = eval_epoch(args, model, valid_dataloader, epoch+1)
            results["valid"].append(new_valid)

        if ((epoch+1) % args.save_epochs == 0):
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch)

        epoch += 1

    if args.save_epochs > 0 and original_epoch != epoch:
        save_checkpoint(args, model, optimizer, lr_scheduler, epoch)

    if args.do_test:
        test_results = eval_epoch(args, model, test_dataloader, epoch+1)
        results["test"].append(test_results)

    return model, results

def main():

    args = get_args(manual_config="/home/showalte/research/prob_seq_queries/archive/experiments/train/amazon.yaml")
    ROOT = os.path.normpath(os.path.join(__file__,"../../../"))
    DEVICE = args.device
    DISABLE_TQDM = args.disable_tqdm
    # text_dict= load_text(args.data_path)
    text_dict= load_amazon_data(args.data_path)
    args.text_dict = text_dict
    print(text_dict['char_to_id'])
    print("====="*10, flush=True)
    # train_dl, val_dl, test_dl = process_text_data(text_dict, args)
    train_dl, val_dl, test_dl = process_amazon_data(text_dict, args)
    args.text_dict['text'] = None #Keep memory small

    model, results = train(
        args,
        train_dl, val_dl,
    )

    save_path =os.path.join(ROOT,"models/amazon/")
    os.makedirs(save_path,exist_ok = True)
    torch.save(model.state_dict(), f"{save_path}/amazon_model.pt")

#################################################################################
#   Main Method
#################################################################################
if __name__ == "__main__":
    main()



