#################################################################################
#
#             Project Title:  Testing sampling algorithms
#             Author:         Sam Showalter
#             Date:           2022.03.31
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
import inspect
import torch
import numpy as np
import torch.nn as nn
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from seq_queries.arguments import get_args
from seq_queries.sample import (
    mc_sample_importance, mc_sample_random_batch, mc_sample_random_list,
    sample_beam_search
)
from seq_queries.model import CausalLM
from seq_queries.data import load_text, process_data


#################################################################################
#   Function-Class Declaration
#################################################################################


#######################################################################
#  Testing functions (Script at bottom)
#######################################################################

def test_mc_sample_random_batch(data_batch, model, num_seqs,seq_len, excluded, num_hists, device):
    seqs = mc_sample_random_batch(data_batch,
                                model.vocab_size,
                                num_seqs,
                                seq_len,
                                excluded = excluded,
                                device = device,
                                )

    assert tuple(seqs.shape) == (num_hists,num_seqs, seq_len),\
        f"Expected random sample batch sequences tensor to be of shape: {(num_hists,num_seqs, seq_len)}, got seqs: {tuple(seqs.shape)}"
    print("Random sampling batch passed test cases")

#######################################################################

def test_mc_sample_random_list(data_list, model, num_seqs,seq_lens, excluded, num_hists, device):
    seqs = mc_sample_random_list(data_list,
                                model.vocab_size,
                                num_seqs,
                                seq_lens,
                                device = device,
                                excluded = excluded)

    assert len(seqs) == num_hists,\
        f"Expected random sample list length of {num_hists}, got seqs: {len(seqs)}"
    for i in range(num_hists):
        assert tuple(seqs[i].shape) == (num_seqs, seq_lens[i]),\
            f"Random sample list had sequence size mismatch: expected ({num_seqs}, {seq_lens[i]}) but got {tuple(seqs[i].shape)}"
    print("Random sampling list passed test cases")

#######################################################################

def test_mc_sample_importance_batch(data_batch, model, num_seqs,seq_len, excluded, num_hists, device, batch = True):
    seqs, probs = mc_sample_importance(model,
                                    data_batch,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_len,
                                    batch = True,
                                    device = device,
                                    excluded = excluded)

    assert tuple(seqs.shape) == (num_hists,num_seqs, seq_len),\
        f"Expected importance sample batch sequences tensor to be of shape: {(num_hists,num_seqs, seq_len)}, got seqs: {tuple(seqs.shape)}"
    assert not probs.isnan().any() and tuple(probs.shape) == (num_hists, num_seqs),\
        f"Expected importance sample batch probabilities tensor to be of shape: {(num_hists,num_seqs)}, got seqs: {tuple(probs.shape)}"
    print("Importance sampling batch passed test cases")

#######################################################################

def test_mc_sample_importance_list(data_list, model, num_seqs,seq_lens, excluded,num_hists, device, batch = False):
    seqs, probs = mc_sample_importance(model,
                                    data_list,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_lens,
                                    device = device,
                                    batch = batch,
                                    excluded = excluded)

    assert len(seqs) == num_hists,\
        f"Expected importance sample list length of {num_hists}, got seqs: {len(seqs)}"
    for i in range(num_hists):
        assert tuple(seqs[i].shape) == (num_seqs, seq_lens[i]),\
            f"Importance sample list had sequence size mismatch: expected ({num_seqs}, {seq_lens[i]}) but got {tuple(seqs[i].shape)}"
        assert not probs[i].isnan().any() and tuple(probs[i].shape) == (num_seqs,),\
            f"Beam search sampling list had probability size mismatch: expected ({num_seqs}) but got {tuple(probs[i].shape)}"
    print("Importance sampling list passed test cases")

#######################################################################

def test_beam_search_batch(data_batch, model, num_seqs,seq_len, excluded,
                            num_hists, device):
    seqs,probs = sample_beam_search(model,
                                    data_batch,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_len,
                                    device = device,
                                    excluded = excluded,
                                    bw_params = {'coverage_type':'fixed_width'},
                                    return_beams=False)

    assert tuple(seqs.shape) == (num_hists,num_seqs, seq_len),\
        f"Expected beam search sample batch sequences tensor to be of shape: {(num_hists,num_seqs, seq_len)}, got seqs: {tuple(seqs.shape)}"
    assert not probs.isnan().any() and tuple(probs.shape) == (num_hists, num_seqs),\
        f"Expected beam search sample batch probabilities tensor to be of shape: {(num_hists,num_seqs)}, got seqs: {tuple(probs.shape)}"
    print("Beam search sampling batch passed test cases")

#######################################################################

def test_beam_search_list_static(data_list, model, num_seqs,seq_lens, excluded,
                                  num_hists, device):
    seqs,probs = sample_beam_search(model,
                                    data_list,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_lens,
                                    device = device,
                                    excluded = excluded,
                                    return_beams = False,
)

    assert len(seqs) == num_hists and len(probs) == num_hists,\
        f"Expected list length of {num_hists}, got seqs: {len(seqs)}, probs: {len(probs)}"
    for i in range(num_hists):
        assert tuple(seqs[i].shape) == (num_seqs, seq_lens[i]),\
            f"Beam search sampling list had sequence size mismatch: expected ({num_seqs}, {seq_lens[i]}) but got {tuple(seqs[i].shape)}"
        assert not probs[i].isnan().any() and tuple(probs[i].shape) == (num_seqs,),\
            f"Beam search sampling list had probability size mismatch: expected ({num_seqs}) but got {tuple(probs[i].shape)}"
    print("Beam search sampling list static passed test cases")

#######################################################################

def test_beam_search_list_dynamic(data_list, model, num_seqs,seq_lens, excluded,
                                  num_hists,device, bw_params = {'coverage_type':'backoff'}):

    seqs,probs,(beams,covs) = sample_beam_search(model,
                                    data_list,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_lens,
                                    device = device,
                                    excluded = excluded,
                                    return_beams = True,
                                    bw_params = bw_params,
                                )

    assert len(seqs) == num_hists and len(probs) == num_hists,\
        f"Expected list length of {num_hists}, got seqs: {len(seqs)}, probs: {len(probs)}"
    for i in range(num_hists):
        # if i < 10:
        #     print(i, seq_lens[i] - data_list[i].shape[0], seqs[i].shape, probs[i].shape, covs[i])
        #     print("-----"*10)
        assert (len(beams[i]) == len(covs[i])),\
                f"Beam search sampling list had beam width/coverage size mismatch: expected beams: {len(beams[i])} coverages: {len(covs[i])}"
        assert (seqs[i].shape[1] == seq_lens[i]),\
            f"Beam search sampling list had sequence size mismatch: expected ({num_seqs}, {seq_lens[i]}) but got {tuple(seqs[i].shape)}"
        assert not probs[i].isnan().any() and probs[i].shape[0] == seqs[i].shape[0],\
            f"Beam search sampling list had probability and/or beam size mismatch: expected ({len(seqs[i].shape[0])}) but got {tuple(probs[i].shape)}"
    print("Beam search sampling list dynamic passed test cases")


#################################################################################
#   Test orchestrator
#################################################################################


def main():
    config_path = "config/testing/sample.yaml"
    args = get_args(manual_config = config_path)
    device = f"cuda:{args.device_num}"
    MODEL_PATH = "models/shakespeare/shakespeare_model.pt"
    DATA_PATH = "data/shakespeare_input.txt"

    # Data Proceessing
    stacks = load_text(DATA_PATH)
    id_to_char = stacks['id_to_char']
    char_to_id = stacks['char_to_id']
    train_stacks, valid_stacks, test_stacks = process_data(stacks, args)

    # Model instantiation
    vocab_size = stacks["vocab_size"]
    model = CausalLM(
        vocab_size=vocab_size,
        embed_dim=args.hidden_size,
        rnn=nn.LSTM(
            input_size=vocab_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            batch_first=True,
            dropout=args.dropout,
            bidirectional=False,
        ),
    )

    # Loads model weights into an already instantiated model
    model.load_state_dict(torch.load(MODEL_PATH,
                                     map_location = torch.device(device)
                          ),
    ); model.to(device)
    model.eval()
    print(f"\nNumber of Parameters = {sum(p.numel() for p in model.parameters())}\n")
    # print(model)

    # General parameters
    sample = next(iter(train_stacks))
    num_hists = len(sample)
    num_seqs = 50
    beam_width_dynamic = 0.50

    # Batch parameters
    hist_len = 10
    seq_len = 15
    data_batch =torch.stack([s[:hist_len] for s in sample],dim =0).cpu()

    #List parameters
    data = []
    hist_lens = [np.random.randint(5,15) for i in range(num_hists)]
    seq_lens = [np.random.randint(18,25) for i in range(num_hists)]
    data_list =[s[:hist_len].cpu() for s,hist_len in zip(sample,hist_lens)]
    coverage_type = 'backoff'
    excluded =[]


    test_mc_sample_random_batch(data_batch,model,num_seqs,seq_len,excluded, num_hists, device)
    test_mc_sample_random_list(data_list,model,num_seqs,seq_lens,excluded, num_hists, device)
    test_mc_sample_importance_batch(data_batch,model,num_seqs,seq_len,excluded,num_hists, device, batch = True)
    test_mc_sample_importance_list(data_list,model,num_seqs,seq_lens,excluded, num_hists, device, batch = False)
    test_beam_search_batch(data_batch,model,num_seqs,seq_len,excluded, num_hists, device)
    test_beam_search_list_static(data_list,model,num_seqs,seq_lens,excluded,num_hists,device)
    test_beam_search_list_dynamic(data_list,model,beam_width_dynamic,seq_lens,excluded,num_hists, device, bw_params = {"coverage_type":"backoff"})
    test_beam_search_list_dynamic(data_list,model,beam_width_dynamic,seq_lens,excluded, num_hists, device, bw_params = {"coverage_type":"interpolate"})
    print("====="*10)
    print("ALL TEST CASES PASSED")
    print("====="*10)

#######################################################################
# Main Method
#######################################################################
if __name__ == "__main__":
     main()




