#################################################################################
#
#             Project Title:  Sampling Method
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
import torch.nn as nn
from tqdm import tqdm
from utils.data import *
from modeling.char import CharLM

#################################################################################
#   Monte carlo random sampling (batch and variable)
#################################################################################

def mc_sample_random_batch(
    histories,
    vocab_size,
    num_seqs,
    total_seq_len,
    excluded = [],
    device = 'cpu',
):
    """Monte-carlo sampling, where each sequence is
    chosen randomly from the vocabulary. Assumes that each
    sequence is of fixed length

    :histories:     torch.Tensor: History tensor or list, (batch, hist_len)
    :vocab_size:    int:          Size of vocabulary
    :num_seqs:      int:          Number of sequences per history
    :total_seq_len: int:          Total sequence length
    :excluded:      List[int]:    List of excluded labels
    :device:        str:          Device on which to do calculations (gpu, cpu)

    :returns:       torch.Tensor: Tensor of (num_hists, num_seqs, seq_len)
    """
    # Set weights to make random choice
    # from only valid tokens
    num_hists, hist_len = histories.shape
    seq_len = total_seq_len - hist_len
    weights = torch.ones(vocab_size)
    weights[excluded] *= 0

    # Build sequences
    seqs = torch.stack([
        torch.cat(
            (
                # (num_seqs, hist_len)
                histories[i].repeat((num_seqs,1)),
                # (num_seqs, seq_len)
                torch.multinomial(
                    weights.repeat((num_seqs,1)),
                    num_samples = seq_len,
                    replacement = True,
                )
            ), dim = 1)
        for i in range(num_hists)
    ], dim = 0)

    # (hum_hists, num_seqs, total_seq_len)
    return seqs

def mc_sample_random_list(
    histories,
    vocab_size,
    num_seqs,
    total_seq_lens,
    excluded = [],
    device = 'cpu',
):
    """Monte-carlo sampling, where each sequence is
    chosen randomly from the vocabulary. Assumes that each
    sequence is of variable length

    :histories:      List[torch.Tensor]: History tensor or list, (batch, hist_len)
    :vocab_size:     int:                Size of vocabulary
    :num_seqs:       int:                Number of sequences per history
    :total_seq_lens: List[int]:          Total sequence length
    :excluded:       List[int]:          List of excluded labels
    :device:         str:                Device on which to do calculations (gpu, cpu)

    :returns:        torch.Tensor:       List of length num_histories, each element with (num_seqs, seq_len_i)

    """
    # Set weights to make random choice
    # from only valid tokens
    num_hists = len(histories)
    hist_lens = [hist.shape[-1] for hist in histories]
    seq_lens = [total_seq_len - hist_len for total_seq_len,hist_len
                in zip(total_seq_lens, hist_lens)]
    weights = torch.ones(vocab_size)
    weights[excluded] *= 0

    # Build sequences
    seqs = [
        torch.cat(
            (
                # (num_seqs, hist_len)
                histories[i].repeat((num_seqs,1)),
                # (num_seqs, seq_len)
                torch.multinomial(
                    weights.repeat((num_seqs,1)),
                    num_samples = seq_lens[i],
                    replacement = True,
                )
            ), dim = 1)
        for i in range(num_hists)
    ]

    # (batch, num_seqs, total_seq_len)
    return seqs

#######################################################################
# Monte carlo importance sampling (batch and variable)
#######################################################################


@torch.no_grad()
def mc_sample_importance_batch(
    model,
    histories,
    vocab_size,
    num_seqs,
    total_seq_len,
    batch_size = 32,
    temperature = 1,
    rnn_args = None,
    excluded = [],
    tqdm_disable = False,
    device = 'cpu',
    eps = 1e-10,
):
    """Monte-carlo sampling with importance sampling
    provided by some model. First duplicates all history
    sequences to have (num_seqs) copies, and then produces
    an importance sampled sequence for each of size total_seq_len.

    :model:         nn.Module:      Torch model utilized for importance sampling
    :histories:     torch.Tensor:   history sequences (num_samples, hist_len)
    :vocab_size:    int:            Size of vocabulary
    :num_seqs:      int:            Number of sequences to generate per history sample
    :total_seq_len: int:            Desired length of entire sequence (history + generated)
    :batch_size:    int:            Not used - can modulate load on GPU
    :temperature:   int:            Temperature for sampling probabilities
    :rnn_args:      Optional[Dict]: Sampling model arguments
    :excluded:      List:           List of indices the model should not sample
    :tqdm_disable:  bool:           Disable tqdm if it clutters output
    :device:        str:            Device on which to do calculations (gpu, cpu)

    :returns:       torch.Tensor:   Sequences tensor of shape (num_hists, num_seqs, seq_len)
    """
    num_hists, hist_len = histories.shape
    seq_len = total_seq_len - hist_len
    legal_vocab = np.array(
        [val for val in np.arange(vocab_size) if val not in excluded]
    )

    # Get probabilities from model
    # (num_histories, vocab)
    probs, _ = model.get_next_probs(
        histories.to(device),
        rnn_args = rnn_args,
        temperature = temperature,
    ); probs += eps

    # Zero out weights on excluded tokens
    probs[:,excluded] = eps/2
    probs /= probs.sum(dim=-1).reshape(-1,1)

    # Log probabilities start (num_hists, num_seqs)
    log_probs = torch.stack([
                # (num_seqs, 1)
                torch.log(probs[i,:].repeat((num_seqs,1)))
        for i in range(num_hists)
     ], dim = 0)

    # Build original sequences
    # With one added term
    # (num_hists, num_seqs, hist_len + 1)
    seqs = torch.stack([
        torch.cat(
            (
                # (num_seqs, hist_len)
                histories[i,:].repeat((num_seqs,1)),
                # (num_seqs, 1)
                torch.multinomial(
                    probs[i,:].repeat((num_seqs,1)),
                    num_samples = 1,
                    replacement = True,
                )
            ), dim = 1)
        for i in range(num_hists)
     ], dim = 0)

    # Select only probabilites that were selected
    # (num_hists, num_seqs)
    log_probs = torch.gather(log_probs, -1,
                    torch.unsqueeze(seqs[:,:,-1],-1))

    # Iterate through to end of sequence,
    # sampling all values, but already got one
    for _ in tqdm(range(1, seq_len, 1),disable = tqdm_disable):

        # Get probabilities from model
        # (batch, num_seqs, vocab)
        probs = torch.stack([
                model.get_next_probs(
                    seqs[i,...].to(device),
                    rnn_args = rnn_args,
                    temperature = temperature,
                )[0] # Only want probabilities, not hidden states
                for i in range(num_hists)
         ], dim = 0)

        # Probabilities need to be squashed
        probs[:,:,excluded] = eps/2
        probs /= probs.sum(dim=-1).reshape(num_hists,num_seqs,1)

        seqs = torch.cat(
                (
                    seqs,
                    # (num_seqs, 1)
                    torch.multinomial(
                        probs.reshape(-1,vocab_size),
                        num_samples = 1,
                        replacement = True,
                    # Reshape to align with tensor
                    ).reshape(num_hists, num_seqs, 1)
                ), dim = -1
        )
        log_probs += torch.gather(
            probs, -1,
            torch.unsqueeze(seqs[:,:,-1],-1),
        )

    # Squeeze out last dimension
    return seqs, torch.squeeze(log_probs,-1)

@torch.no_grad()
def mc_sample_importance_list(
    model,
    histories,
    vocab_size,
    num_seqs,
    total_seq_lens,
    batch_size = 32,
    temperature = 1,
    rnn_args = None,
    excluded = [],
    tqdm_disable = False,
    device = 'cpu',
    eps = 1e-10,
):
    """Monte-carlo sampling with importance sampling
    provided by some model. First duplicates all history
    sequences to have (num_seqs) copies, and then produces
    an importance sampled sequence for each of size total_seq_len.
    Assumes history and total sequence lengths can differ by sample.

    - fix tolerance
    - Get the importance sampled probabilities
    - normalize probabilities

    :model:          nn.Module:          Torch model utilized for importance sampling
    :histories:      List[torch.Tensor]: history sequences list where each element is (hist_len_i)
    :vocab_size:     int:                Size of vocabulary
    :num_seqs:       int:                Number of sequences to generate per history sample
    :total_seq_lens: List[int]:          Desired length of entire sequence (history + generated) for each element
    :batch_size:     int:                Not used - can modulate load on GPU
    :temperature:    int:                Temperature for sampling probabilities
    :rnn_args:       Optional[Dict]:     Sampling model arguments
    :excluded:       List[int]:          List of indices the model should not sample
    :tqdm_disable:   bool:               Disable tqdm if it clutters output
    :device:         str:                Device on which to do calculations (gpu, cpu)

    :returns:        List[torch.Tensor]: List of tensors, each of shape (num_seqs, seq_len_i)
    """
    num_hists = len(histories)
    hist_lens = [h.shape[-1] for h in histories]
    seq_lens = [total_seq_len_i - hist_len for total_seq_len_i,hist_len
                in zip(total_seq_lens,hist_lens)]
    excluded = set(excluded)
    legal_vocab = np.array(
        [val for val in np.arange(vocab_size) if val not in excluded]
    )
    excluded = list(excluded)

    # Get probabilities from model
    # (num_histories, vocab)
    probs= [
            model.get_next_probs(
                histories[i].reshape(1, hist_lens[i]).to(device),
                rnn_args = rnn_args,
                temperature = temperature,
            )[0] + eps
        for i in range(num_hists)
    ]

    # Zero out weights on excluded tokens
    for i in range(num_hists):
        probs[i][:,excluded] *= eps/2
        probs[i] /= probs[i].sum(dim=-1).reshape(-1,1)

    # Log probabilities start (num_hists, num_seqs)
    log_probs = [ # (num_seqs, vocab)
        torch.log(probs[i].repeat((num_seqs,1)))
        for i in range(num_hists)
     ]

    # Build original sequences
    seqs = [
        torch.cat(
            (
                # (num_seqs, hist_len)
                histories[i].repeat((num_seqs,1)),
                # (num_seqs, 1)
                torch.multinomial(
                    probs[i].repeat((num_seqs,1)),
                    num_samples = 1,
                    replacement = True,
                )
            ), dim = 1)
        for i in range(num_hists)
     ]

    # Select only probabilites that were selected
    # List[(num_seqs)]
    log_probs = [
        torch.gather(log_probs[i], -1,
            torch.unsqueeze(seqs[i][:,-1],-1))
        for i in range(num_hists)
    ]

    # Iterate through to end of sequence,
    # sampling all values, but already got one
    for pos in tqdm(range(1, max(seq_lens), 1),
                    disable = tqdm_disable):
        # Get probabilities from model
        # (batch, num_seqs, vocab)
        probs = [
                model.get_next_probs(
                    seqs[i].to(device),
                    rnn_args = rnn_args,
                    temperature = temperature,
                )[0] + eps if (pos < seq_lens[i]) else None
            for i in range(num_hists)
         ]

        # Add to sequences only if they aren't done
        for i in range(num_hists):
            if probs[i] is not None:
                probs[i][:,excluded] = eps/2
                addition = torch.multinomial(
                    probs[i],
                    num_samples = 1,
                    replacement = True,
                )

                seqs[i] = torch.cat(
                    (seqs[i], addition), dim = -1
                )

                log_probs[i] += torch.gather(
                    probs[i], -1,
                    torch.unsqueeze(seqs[i][:,-1],-1),
                )

    log_probs = [torch.squeeze(log_prob,-1) for log_prob in log_probs]
    return seqs, log_probs


#######################################################################
# Beam Search helper functions
#######################################################################

@torch.no_grad()
def get_beam_width(
    curr_beam, prob,
    percentile = False,
):
    """TODO: Docstring for _get_beam_width.

    :beam_cov: TODO
    :prob: TODO
    :: TODO
    :returns: TODO

    """
    if not percentile:
        return curr_beam

    cum_probs = torch.cumsum(prob.flatten(),0)
    beam_width = torch.argwhere(
        cum_probs >= curr_beam
    ).flatten()[0] + 1

    return beam_width

def prepare_beams(
    beam_widths,
    num_hists,
    seq_lens,
):
    """TODO: Docstring for prepare_beams.

    :beam_width: TODO
    :returns: TODO

    """
    percentile = False
    if isinstance(beam_widths,int):
        orig_beam_widths = np.array([beam_widths]*num_hists)
        curr_beams = copy.deepcopy(orig_beam_widths)
    elif isinstance(beam_widths, float):
        percentile = True
        orig_beam_widths = np.array([beam_widths]*num_hists)
        curr_beams = np.array(
            [np.power(beam_widths, 1/seq_lens[i]) for i in range(num_hists)]
        )
    elif isinstance(beam_widths, list):
        assert len(beam_widths) == num_hists,"Histories and beam widths not aligned"
        orig_beam_widths = np.array([beam_widths]*num_hists)
        if isinstance(beam_widths[0],float):
            percentile = True
            curr_beams = np.array(
                [np.pow(beam_widths, 1/seq_lens[i]) for i in range(num_hists)]
            )
        else: curr_beams = copy.deepcopy(orig_beam_widths)
    else:
        assert False,"Beam data type must be int, float or List[int/float]"
    return curr_beams, orig_beam_widths, percentile

#######################################################################
# Beam Search (batch and variable)
#######################################################################

@torch.no_grad()
def sample_beam_search_batch(
    model,
    histories,
    vocab_size,
    beam_width,
    total_seq_len,
    batch_size = 32,
    temperature = 1,
    rnn_args = None,
    excluded = [],
    disable_tqdm = False,
    eps = 1e-10,
    device = 'cpu',
):
    """Beam search where thes histories and the
    sequence lengths all have the same size.

    :model:         nn.Module:      Torch model utilized for importance sampling
    :histories:     torch.Tensor:   history sequences (num_samples, hist_len)
    :vocab_size:    int:            Size of vocabulary
    :beam_width:    int:            Number of sequences to generate per history sample
    :total_seq_len: int:            Desired length of entire sequence (history + generated)
    :batch_size:    int:            Not used - can modulate load on GPU
    :temperature:   int:            Temperature for sampling probabilities
    :rnn_args:      Optional[Dict]: Sampling model arguments
    :excluded:      List:           List of indices the model should not sample
    :tqdm_disable:  bool:           Disable tqdm if it clutters output
    :device:        str:            Device on which to do calculations (gpu, cpu)

    :returns:       torch.Tensor:   Sequences tensor of shape (num_hists, num_seqs, seq_len)
    """

    # Excluded labels
    num_hists = len(histories)
    seq_len = total_seq_len - hist_len
    excluded = set(excluded)
    legal_vocab = np.array(
        [val for val in np.arange(vocab_size) if val not in excluded]
    )
    excluded = list(excluded)

    # Get probabilities from model for each history
    # (num_hists, vocab)
    probs = [
        model.get_next_probs(
            histories[i].reshape(1, hist_len).to(device),
            rnn_args = rnn_args,
            temperature = temperature,
        )[0] + eps for i in range(num_hists)
    ]

    # Zero out weights on excluded tokens
    for i in range(num_hists):
        probs[i][:,excluded] = eps/2
        probs[i] /= probs[i].sum(dim=1)

    # Top probabilities
    # (num_hists, vocab)
    sorted_probs_inds = [
        torch.sort(
            probs[i].flatten(), dim = -1,
            descending = True
        ) for i in range(num_hists)
    ]

    # Create the logarithm here to seed sum
    sorted_probs_inds = [
        (torch.log(sorted_prob), sorted_ind)
        for sorted_prob, sorted_ind in sorted_probs_inds
    ]

    seqs = [
        torch.cat(
            (
                # (beam_width, hist_len)
                histories[i].reshape(1, hist_len).repeat(
                    # In case the beam width is > vocab_size
                    (min(beam_width,vocab_size),1)),
                # (beam_width, 1)
                sorted_probs_inds[i][1][:beam_width].reshape(-1,1)
            ), dim = 1
        ) for i in range(num_hists)
    ]

    # Iterate through all sequences
    for pos in tqdm(range(1,seq_len,1),disable = disable_tqdm):

        # Get new probabilities
        # List[(beam_width, vocab)] len num_hists
        new_probs = [
            model.get_next_probs(
                seqs[i].to(device),
                rnn_args = rnn_args,
                temperature = temperature,
            )[0] + eps for i in range(num_hists)
        ]

        # Knock down excluded probabilities
        for i in range(num_hists):
            # Make sure excluded tokens are never chosen
            new_probs[i][:,excluded] = eps/2
            new_probs[i] /= new_probs[i].sum(dim=-1).reshape(-1,1)

        # # Add together probabilities to make new measures
        # # List[(beam_width, vocab)] len num_hists
        probs = [ # (beam_width) + (beam_width, vocab)
            sorted_prob_ind[0][:beam_width].reshape(-1,1) + torch.log(new_prob)
            for sorted_prob_ind, new_prob in zip(sorted_probs_inds, new_probs)
        ]

        # Need a way to zero out the probabilities of each new check
        # Top probabilities
        # List[(beam_width, vocab)] len num_hists
        sorted_probs_inds = [
            torch.sort(
                # (beam_width x vocab)
                probs[i].flatten(), dim = -1,
                descending = True,
            ) for i in range(num_hists)
        ]

        # Match sequences with their next best token
        seq_inds = [(sorted_probs_inds[i][1][:beam_width] // vocab_size) for i in range(num_hists)]
        best_tokens = [(sorted_probs_inds[i][1][:beam_width] % vocab_size).long() for i in range(num_hists)]

        # (beam_width, curr_seq_len+1)
        # Need to expand this if the size is growing
        seqs = [
            torch.cat(
                (
                    # (beam_width, curr_seq_len)
                    seqs[i][seq_inds[i], :],
                    # (beam_width,1)
                    best_tokens[i].reshape(-1,1),
                ), dim = 1
            ) for i in range(num_hists)
        ]

    # Get only the most probable tokens and store probabilities
    # (num_hists, beam_width, total_seq_len) and (num_hists, beam_width) for outputs
    log_probs = [probs[i][seq_inds[i], best_tokens[i]].flatten() for i in range(num_hists)]
    return torch.stack(seqs, dim = 0), torch.stack(log_probs,dim =0)


@torch.no_grad()
def sample_beam_search_list(
    model,
    histories,
    vocab_size,
    beam_widths,
    total_seq_lens,
    batch_size = 32,
    temperature = 1,
    rnn_args = None,
    excluded = [],
    disable_tqdm = False,
    device = 'cpu',
    eps = 1e-10,
    return_beams = True,
):
    """Beam search where thes histories and the
    sequence lengths may have different sizes.

    :model:          nn.Module:          Torch model utilized for importance sampling
    :histories:      List[torch.Tensor]: history sequences list where each element is (hist_len_i)
    :vocab_size:     int:                Size of vocabulary
    :beam_width:     int or List[int]:                Number of sequences to generate per history sample
    :total_seq_lens: List[int]:          Desired length of entire sequence (history + generated) for each element
    :batch_size:     int:                Not used - can modulate load on GPU
    :temperature:    int:                Temperature for sampling probabilities
    :rnn_args:       Optional[Dict]:     Sampling model arguments
    :excluded:       List[int]:          List of indices the model should not sample
    :tqdm_disable:   bool:               Disable tqdm if it clutters output
    :device:         str:                Device on which to do calculations (gpu, cpu)

    :returns:        List[torch.Tensor]: List of tensors, each of shape (num_seqs, seq_len_i)
    """

    # Excluded labels
    num_hists = len(histories)
    hist_lens = [h.shape[-1] for h in histories]
    seq_lens = [total_seq_len_i - hist_len for total_seq_len_i,hist_len
                in zip(total_seq_lens,hist_lens)]
    curr_beams,orig_beams, percentile = prepare_beams(beam_widths,num_hists,seq_lens)
    excluded = set(excluded)
    legal_vocab = np.array(
        [val for val in np.arange(vocab_size) if val not in excluded]
    )
    excluded = list(excluded)

    # Get probabilities from model for each history
    # (num_hists, vocab)
    probs = [
        model.get_next_probs(
            histories[i].reshape(1, hist_lens[i]).to(device),
            rnn_args = rnn_args,
            temperature = temperature,
        )[0] + eps for i in range(num_hists)
    ]


    # Zero out weights on excluded tokens
    for i in range(num_hists):
        probs[i][:,excluded] = eps/2
        probs[i] /= probs[i].sum(axis=1).reshape(-1,1)

    # Top probabilities
    # (num_hists, vocab)
    sorted_probs_inds = [
        torch.sort(
            probs[i].flatten(), dim = -1,
            descending = True
        ) for i in range(num_hists)
    ]; sorted_probs = [prob_inds[0] for prob_inds in sorted_probs_inds]

    # Initial seqs
    # List[(beam_width, hist_len + 1)] len num_hists
    beam_widths = [
        get_beam_width(
            curr_beam, prob,
            percentile=percentile,
        ) for curr_beam,prob in
        zip(curr_beams, probs)
    ]; all_beam_widths = [[bw] for bw in beam_widths]

    # Update current beam coverage if percentile used
    if percentile:
        curr_beams = [curr_beams[i]*orig_beams[i] for i in range(num_hists)]

    # Create the logarithm here to seed sum
    sorted_probs_inds = [
        (torch.log(sorted_prob), sorted_ind)
        for sorted_prob, sorted_ind in sorted_probs_inds
    ]

    # Initial seqs
    # List[(beam_width, hist_len + 1)] len num_hists
    seqs = [
        torch.cat(
            (
                # (beam_width, hist_len)
                histories[i].reshape(1, hist_lens[i]).repeat(
                    # In case the beam width is > vocab_size
                    (min(beam_widths[i],vocab_size),1)),
                # (beam_width, 1)
                sorted_probs_inds[i][1][:beam_widths[i]].reshape(-1,1)
            ), dim = 1
        ) for i in range(num_hists)
    ]

    # Iterate through all sequences
    for pos in tqdm(range(1,max(seq_lens)+1,1),disable = disable_tqdm):

        # Get new probabilities
        # List[(beam_width, vocab)] len num_hists
        new_probs = [
            model.get_next_probs(
                seqs[i].to(device),
                rnn_args = rnn_args,
                temperature = temperature,
            )[0] + eps if (pos < seq_lens[i]) else None
            for i in range(num_hists)
        ]

        # Knock down excluded probabilities
        for i in range(num_hists):
            if new_probs[i] is not None:
                new_probs[i][:,excluded] = eps/2
                new_probs[i][:,excluded] /= new_probs[i].sum(dim = -1).reshape(-1,1)

        # Add together probabilities to make new measures
        # List[(beam_width, vocab)] len num_hists
        # Store all probabilities even from earlier sequences
        probs = [ # (beam_width) + (beam_width, vocab)
            sorted_probs_inds[i][0][:all_beam_widths[i][-1]].reshape(-1,1) + torch.log(new_probs[i])
                 if new_probs[i] is not None else sorted_probs_inds[i][0][:all_beam_widths[i][-1]]
            for i in range(num_hists)
        ]

        # Need a way to zero out the probabilities of each new check
        # Top probabilities
        # List[(beam_width, vocab)] len num_hists
        for i in range(num_hists):
            if new_probs[i] is not None:
                sorted_probs_inds[i] = \
                    torch.sort(
                        # (beam_width x vocab)
                        probs[i].flatten(), dim = -1,
                        descending = True,
                    )
                sorted_probs = torch.exp(sorted_probs_inds[i][0])
                sorted_probs /= sorted_probs.sum()
                assert not sorted_probs.isnan().any(),"Nans in sorted probabilities"

                beam_width = get_beam_width(
                    curr_beams[i], sorted_probs,
                    percentile=percentile,
                ); all_beam_widths[i].append(beam_width)
                if percentile: curr_beams[i]*=orig_beams[i] #Must update individually

                # Match sequences with their next best token
                tokens = sorted_probs_inds[i][1][:beam_width]
                seq_inds = tokens // vocab_size
                best_tokens = tokens % vocab_size

                # (beam_width, curr_seq_len+1)
                # Need to expand this if the size is growing
                sorted_probs_inds[i] = (
                    sorted_probs_inds[i][0][seq_inds],
                    best_tokens
                )
                seqs[i] = \
                    torch.cat(
                        (
                            # (beam_width, curr_seq_len)
                            seqs[i][seq_inds, :],
                            # (beam_width,1)
                            best_tokens.reshape(-1,1),
                        ), dim = 1
                    )
    # Seqs = List[(beam_width, total_seq_len_i)]
    # Probs = List[(beam_width)]
    # Return sequences and log probabilities
    if return_beams:
        return seqs, probs, all_beam_widths
    return seqs, probs


#################################################################################
#   TESTING
#################################################################################


if __name__ == "__main__":

    ROOT = os.path.normpath(__file__)
    DEVICE = "cpu"
    DISABLE_TQDM = True
    MODEL_PATH = "models/shakespeare/shakespeare_model.pt"
    DATA_PATH = "data/shakespeare_input.txt"

    # Data Proceessing
    stacks = load_text(DATA_PATH)
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

    # Loads model weights into an already instantiated model
    model.load_state_dict(torch.load(MODEL_PATH,
                                     map_location = torch.device(DEVICE)
                          ),
    )
    model.eval()
    print(f"\nNumber of Parameters = {sum(p.numel() for p in model.parameters())}\n")
    # print(model)

    # General parameters
    sample = next(iter(train_stacks))
    num_hists = len(sample)
    num_seqs = 50
    beam_width_dynamic = 0.92

    # Batch parameters
    hist_len = 10
    seq_len = 20
    data_batch =torch.stack([s[:hist_len] for s in sample],dim =0)

    #List parameters
    data = []
    hist_lens = [np.random.randint(5,15) for i in range(num_hists)]
    seq_lens = [np.random.randint(18,25) for i in range(num_hists)]
    data_list =[s[:hist_len] for s,hist_len in zip(sample,hist_lens)]
    excluded =[model.vocab_size - 3,model.vocab_size - 2]

    #######################################################################
    #######################################################################

    def test_mc_sample_random_batch(data_batch, model, num_seqs,seq_lens, excluded):
        seqs = mc_sample_random_batch(data_batch,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_len,
                                    excluded = excluded)

        assert tuple(seqs.shape) == (num_hists,num_seqs, seq_len),\
            f"Expected random sample batch sequences tensor to be of shape: {(num_hists,num_seqs, seq_len)}, got seqs: {tuple(seqs.shape)}"
        print("Random sampling batch passed test cases")

    #######################################################################


    def test_mc_sample_random_list(data_list, model, num_seqs,seq_lens, excluded):
        seqs = mc_sample_random_list(data_list,
                                    model.vocab_size,
                                    num_seqs,
                                    seq_lens,
                                    excluded = excluded)

        assert len(seqs) == num_hists,\
            f"Expected random sample list length of {num_hists}, got seqs: {len(seqs)}"
        for i in range(num_hists):
            assert tuple(seqs[i].shape) == (num_seqs, seq_lens[i]),\
                f"Random sample list had sequence size mismatch: expected ({num_seqs}, {seq_lens[i]}) but got {tuple(seqs[i].shape)}"
        print("Random sampling list passed test cases")

    #######################################################################

    def test_mc_sample_importance_batch(data_batch, model, num_seqs,seq_lens, excluded):
        seqs, probs = mc_sample_importance_batch(model,
                                        data_batch,
                                        model.vocab_size,
                                        num_seqs,
                                        seq_len,
                                        excluded = excluded)

        assert tuple(seqs.shape) == (num_hists,num_seqs, seq_len),\
            f"Expected importance sample batch sequences tensor to be of shape: {(num_hists,num_seqs, seq_len)}, got seqs: {tuple(seqs.shape)}"
        assert not probs.isnan().any() and tuple(probs.shape) == (num_hists, num_seqs),\
            f"Expected importance sample batch probabilities tensor to be of shape: {(num_hists,num_seqs)}, got seqs: {tuple(probs.shape)}"
        print("Importance sampling batch passed test cases")

    #######################################################################

    def test_mc_sample_importance_list(data_list, model, num_seqs,seq_lens, excluded):
        seqs, probs = mc_sample_importance_list(model,
                                        data_list,
                                        model.vocab_size,
                                        num_seqs,
                                        seq_lens,
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

    def test_beam_search_batch(data_batch, model, num_seqs,seq_len, excluded):
        seqs,probs = sample_beam_search_batch(model,
                                        data_batch,
                                        model.vocab_size,
                                        num_seqs,
                                        seq_len,
                                        excluded = excluded)

        assert tuple(seqs.shape) == (num_hists,num_seqs, seq_len),\
            f"Expected beam search sample batch sequences tensor to be of shape: {(num_hists,num_seqs, seq_len)}, got seqs: {tuple(seqs.shape)}"
        assert not probs.isnan().any() and tuple(probs.shape) == (num_hists, num_seqs),\
            f"Expected beam search sample batch probabilities tensor to be of shape: {(num_hists,num_seqs)}, got seqs: {tuple(probs.shape)}"
        print("Beam search sampling batch passed test cases")

    #######################################################################

    def test_beam_search_list_static(data_batch, model, num_seqs,seq_lens, excluded):
        seqs,probs = sample_beam_search_list(model,
                                        data_list,
                                        model.vocab_size,
                                        num_seqs,
                                        seq_lens,
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

    def test_beam_search_list_dynamic(data_batch, model, num_seqs,seq_lens, excluded):
        seqs,probs,beam_widths = sample_beam_search_list(model,
                                        data_list,
                                        model.vocab_size,
                                        num_seqs,
                                        seq_lens,
                                        excluded = excluded,
                                        return_beams = True,
    )

        assert len(seqs) == num_hists and len(probs) == num_hists,\
            f"Expected list length of {num_hists}, got seqs: {len(seqs)}, probs: {len(probs)}"
        for i in range(num_hists):
            print(seqs[i].shape, probs[i].shape, len(beam_widths[i]))
            assert (seqs[i].shape[1] == seq_lens[i]),\
                f"Beam search sampling list had sequence size mismatch: expected ({num_seqs}, {seq_lens[i]}) but got {tuple(seqs[i].shape)}"
            assert not probs[i].isnan().any() and probs[i].shape[0] == seqs[i].shape[0],\
                f"Beam search sampling list had probability and/or beam size mismatch: expected ({len(beam_widths[i])}) but got {tuple(probs[i].shape)}"
        print("Beam search sampling list dynamic passed test cases")

    #######################################################################

    # test_mc_sample_random_batch(data_batch,model,num_seqs,seq_len,excluded)
    # test_mc_sample_random_list(data_list,model,num_seqs,seq_lens,excluded)
    # test_mc_sample_importance_batch(data_batch,model,num_seqs,seq_len,excluded)
    # test_mc_sample_importance_list(data_list,model,num_seqs,seq_lens,excluded)
    # test_beam_search_batch(data_batch,model,num_seqs,seq_len,excluded)
    # test_beam_search_list_static(data_list,model,num_seqs,seq_lens,excluded)
    test_beam_search_list_dynamic(data_list,model,beam_width_dynamic,seq_lens,excluded)
    print("====="*10)
    print("ALL TEST CASES PASSED")
    print("====="*10)










