import os
import math
import random
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_ as clip_grad

from collections import defaultdict
from tqdm import tqdm

from seq_queries.optim import get_optimizer, get_lr_scheduler
from seq_queries.utils import print_log
from seq_queries.model import get_model
from seq_queries.data import load_text, process_data
from seq_queries.arguments import get_args


def forward_pass(args, batch, model):
    if args.cuda:
        batch = {k:v.to(args.device) for k,v in batch.items()}

    # Forward Pass
    return model.graded_forward(**batch)

def backward_pass(args, loss, model, optimizer, lr_scheduler):
    optimizer.zero_grad()
    loss.backward()
    clip_grad(parameters=model.parameters(), max_norm=args.grad_clip, norm_type=2)
    optimizer.step()
    lr_scheduler.step()

def train_step(args, model, optimizer, lr_scheduler, batch):
    loss_results, forward_results = forward_pass(args, batch, model)
    backward_pass(args, loss_results["loss"], model, optimizer, lr_scheduler)

    return loss_results

def train_epoch(args, model, optimizer, lr_scheduler, dataloader, epoch_number):
    model.train()

    avg_losses = {
        "loss":0.0,
    }
    data_len = len(dataloader)

    training_pbar = tqdm(dataloader)
    for i, batch in enumerate(training_pbar):
        output = train_step(args, model, optimizer, lr_scheduler, batch)

        for key in avg_losses.keys():
            avg_losses[key] = (avg_losses[key]*i + output[key].item()) / (i+1)  # calculate average this way to always have an up to date, scale adjusted estimate
        if (i+1) % args.log_interval == 0:
            training_pbar.set_description("[T] E={}, {}".format(epoch_number, ", ".join(["{}={:.4f}".format(k,v) for k,v in avg_losses.items()])))
    training_pbar.set_description("[T] E={}, {}".format(epoch_number, ", ".join(["{}={:.4f}".format(k,v) for k,v in avg_losses.items()])))

    return avg_losses

@torch.no_grad()
def eval_step(args, model, batch):
    return forward_pass(args, batch, model)

def eval_epoch(args, model, dataloader, epoch_number):
    model.eval()

    avg_losses = {
        "loss":0.0,
    }
    data_len = len(dataloader)

    validation_pbar = tqdm(dataloader)
    for i, batch in enumerate(validation_pbar):
        output = eval_step(args, model, batch)

        for key in avg_losses.keys():
            avg_losses[key] = (avg_losses[key]*i + output[key].item()) / (i+1)  # calculate average this way to always have an up to date, scale adjusted estimate
        if (i+1) % args.log_interval == 0:
            validation_pbar.set_description("[V] E={}, {}".format(epoch_number, ", ".join(["{}={:.4f}".format(k,v) for k,v in avg_losses.items()])))
    validation_pbar.set_description("[V] E={}, {}".format(epoch_number, ", ".join(["{}={:.4f}".format(k,v) for k,v in avg_losses.items()])))

    return avg_losses

def set_random_seed(args):
    """Set random seed for reproducibility."""

    seed = args.seed

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

def setup_model_and_optim(args, epoch_len):
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_lr_scheduler(optimizer, args, epoch_len)

    return model, optimizer, lr_scheduler

def save_checkpoint(args, model, optimizer, lr_scheduler, epoch):
    # Create folder if not already created
    folder_path = args.checkpoint_path
    folders = folder_path.split("/")
    for i in range(len(folders)):
        if folders[i] == "":
            continue
        intermediate_path = "/".join(folders[:i+1])
        if not os.path.exists(intermediate_path):
            os.mkdir(intermediate_path)

    final_path = "{}/model_{:03d}.pt".format(folder_path.rstrip("/"), epoch)
    if os.path.exists(final_path):
        os.remove(final_path)
    torch.save(model.state_dict(), final_path)
    print_log("Saved model at {}".format(final_path))

def load_checkpoint(args, model):
    folder_path = args.checkpoint_path
    if not os.path.exists(folder_path):
        print_log(f"Checkpoint path [{folder_path}] does not exist.")
        return 0

    print_log(f"Checkpoint path [{folder_path}] does exist.")
    files = [f for f in os.listdir(folder_path) if ".pt" in f]
    if len(files) == 0:
        print_log("No .pt files found in checkpoint path.")
        return 0

    latest_model = sorted(files)[-1]
    file_path = "{}/{}".format(folder_path.rstrip("/"), latest_model)

    if not os.path.exists(file_path):
        print_log(f"File [{file_path}] not found.")
        return 0

    model.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
    if args.cuda:
        #model.cuda(torch.cuda.current_device())
        model.to(args.device)
    print_log("Loaded model from {}".format(file_path))
    return int(latest_model.replace("model_", "").replace(".pt", "")) + 1

def report_model_stats(model):
    total = 0 

    for name, param in model.named_parameters():
        total += param.numel()

    print_log()
    print_log("Total Parameter Count:", total)
    print_log()

def main(args):
    print_log("Setting seed.")
    set_random_seed(args)

    print_log("Setting up dataloaders.")
    text_dict = load_text(args.data_path)
    train_dataloader, valid_dataloader, test_dataloader = process_data(text_dict, args)
    
    print_log("Setting up model, optimizer, and learning rate scheduler.")
    model, optimizer, lr_scheduler = setup_model_and_optim(args, len(train_dataloader))

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

    return results

if __name__ == "__main__":
    print_log("Getting arguments.")
    args = get_args()
    results = main(args)
    print_log("\n\nResults:", '\n', results)
