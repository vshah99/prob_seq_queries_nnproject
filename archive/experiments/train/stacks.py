#################################################################################
#
#             Project Title:  Train Stacks LM
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os

import torch
import torch.nn as nn

from utils.data import load_text, process_data
from utils.train import train
from modeling.char import CharLM
#################################################################################
#   Function-Class Declaration
#################################################################################

def main():
    ROOT = os.path.normpath(os.path.join(__file__,"../../../"))
    DEVICE = "cuda:1"
    DISABLE_TQDM = True

    # Data Proceessing
    stacks = load_text("data/stacks.tex")
    train_stacks, valid_stacks, test_stacks = process_data(
        text_dict=stacks,
        batch_size=64,
        seq_len=100,
        dev=DEVICE,
        splits=(0.9, 0.05, 0.05),  # Split percentages for (Train, Validation, Test)
    )

    # Hyperparameters
    embed_dim = 128
    num_layers = 2
    dropout = 0.3

    # Model instantiation
    vocab_size = stacks["vocab_size"]
    model = CharLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        rnn=nn.LSTM(
            input_size=vocab_size,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        ),
        dev=DEVICE,
        lr=1e-2,  #1e-3,
    )
    print("Number of Parameters =", sum(p.numel() for p in model.parameters()))
    print(model)

    train(
        model=model,
        epochs=20,
        train_data=train_stacks,
        valid_data=valid_stacks,
        validation_freq=0.1,
        disable_tqdm = DISABLE_TQDM,
    )

    save_path =os.path.join(ROOT,"models/stacks/")
    os.makedirs(save_path,exist_ok = True)
    torch.save(model.state_dict(), f"{save_path}/stacks_model.pt")

#################################################################################
#   Main Method
#################################################################################



