#################################################################################
#
#             Project Title:  Train shakespeare LM
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
    DEVICE = "cuda:0"
    DISABLE_TQDM = True

    # Data Processing
    shakes = load_text("data/shakespeare_input.txt")
    train_shakes, valid_shakes, test_shakes = process_data(
        text_dict=shakes,
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
    vocab_size = shakes["vocab_size"]
    model = CharLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        rnn=nn.LSTM(  # can swap this out with nn.RNN, nn.GRU, etc.
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
    print(model,flush=True)

    train(
        model=model,
        epochs=20,
        train_data=train_shakes,
        valid_data=valid_shakes,
        validation_freq=0.1,
        disable_tqdm = DISABLE_TQDM,
    )

    save_path =os.path.join(ROOT,"models/shakespeare/")
    os.makedirs(save_path,exist_ok = True)
    torch.save(model.state_dict(), f"{save_path}/shakespeare_model.pt")

#################################################################################
#   Main Method
#################################################################################



