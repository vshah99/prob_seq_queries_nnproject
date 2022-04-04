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
import pickle as pkl

import numpy as np
import random
import torch
import torch.nn as nn

from tqdm import tqdm
from .data import load_text, process_data
from .model import CausalLM, MaskedLM

from .arguments import get_args


#################################################################################
#   Evaluate sampler
#################################################################################

def evaluate_seq_query_probs(
    model,
    sample_lists,
    excluded_tokens,
    seq_lens,
    sample_prob_lists = None,
    device = "cuda:0",
    excluded = [],):
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens]*len(sample_lists)
    if sample_prob_lists is None:
        sample_prob_lists = [None]*len(sample_lists)

    estimates = [
        _evaluate_seq_query_prob(
            model,
            sample_lists[i],
            seq_lens[i],
            sample_prob_lists[i],
            device=device,
            excluded=excluded,
        ) for i in range(len(sample_lists))
    ]

    return estimates


def _evaluate_seq_query_prob(
    model,
    samples,
    seq_len,
    sample_probs = None,
    device = "cuda:0",
    excluded = [],
):
    """Evaluate samples from model for a sequential query.
    This could be from a monte carlo sample or from an
    importance sampled sequence

    :model: TODO
    :samples: TODO
    :excluded_tokens: TODO
    :sample_type: TODO
    :: TODO
    :returns: TODO

    """
    probs = model.get_next_probs(
        samples,
        rnn_args = None,
        temperature = None,
        device = device,
    )

    assert tuple(probs.shape) == tuple(sample_probs.shape),\
        f"Probabilities are of different shapes: probs: {probs.shape} sampled: {sample_probs.shape}"
    if sample_probs is not None:
        log_probs = torch.log(probs) - sample_probs

    norm_const = (model.vocab_size - len(excluded))**(seq_len)/probs.shape[0]
    estimate = norm_const* torch.exp(log_probs).sum()
    return estimate


#################################################################################
#   Monte carlo random sampling (batch and variable)
#################################################################################


def mc_sample_random_batch(
    histories,
    vocab_size,
    num_seqs,
    total_seq_lens,
    excluded = [],
    device = 'cpu',
    **kwargs,
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
    seq_len = total_seq_lens - hist_len
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
    return seqs, None, None

def mc_sample_random_list(
    histories,
    vocab_size,
    num_seqs,
    total_seq_lens,
    model = None,
    excluded = [],
    device = 'cpu',
    **kwargs,
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
    return seqs, None, None

#######################################################################
# Importance sampling helper functions
#######################################################################

def importance_sampling_inner_loop(
    model, seqs, log_probs,
    sorted_states,
    num_hists, seq_lens,
    device, temperature,
    iter_range, eps,
    excluded,
    disable_tqdm = False,
):
    """TODO: Docstring for importance_sampling_inner_loop.

    :model: TODO
    :returns: TODO

    """

    # Iterate through to end of sequence,
    # sampling all values, but already got one
    for pos in tqdm(iter_range, disable = disable_tqdm):
        # Get probabilities from model
        # (batch, num_seqs, vocab)
        new_probs_states = [
                None if (pos >= seq_lens[i]) else
                model.get_next_probs(
                    seqs[i][...,-1].reshape(-1,1),
                    rnn_args = sorted_states[i],
                    temperature = temperature,
                    device = device,
                ) for i in range(num_hists)
        ]; new_probs,new_states = list(zip(*[
            (new_prob_state[0] + eps, new_prob_state[1])
            if new_prob_state is not None else (None,None)
            for new_prob_state in new_probs_states]))
        probs, states = list(new_probs), list(new_states)

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
                    torch.log(probs[i]), -1,
                    torch.unsqueeze(seqs[i][:,-1],-1),
                )

    log_probs = [torch.squeeze(log_prob,-1) for log_prob in log_probs]
    return seqs, log_probs

#######################################################################
# Monte carlo importance sampling (batch and variable)
#######################################################################



@torch.no_grad()
def mc_sample_importance(
    histories,
    vocab_size,
    num_seqs,
    total_seq_lens,
    model = None,
    temperature = 1,
    rnn_args = None,
    excluded = [],
    tqdm_disable = False,
    device = 'cpu',
    batch = True,
    eps = 1e-10,
    **kwargs,
):
    """Monte-carlo sampling with importance sampling
    provided by some model. First duplicates all history
    sequences to have (num_seqs) copies, and then produces
    an importance sampled sequence for each of size total_seq_len.
    Assumes history and total sequence lengths can differ by sample.

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
    if isinstance(total_seq_lens, int): total_seq_lens = [total_seq_lens]*num_hists
    seq_lens = [total_seq_len_i - hist_len for total_seq_len_i,hist_len
                in zip(total_seq_lens,hist_lens)]
    excluded = set(excluded)
    legal_vocab = np.array(
        [val for val in np.arange(vocab_size) if val not in excluded]
    ); excluded = list(excluded)

    # Get probabilities from model
    # (num_histories, vocab)
    probs_states = [
        model.get_next_probs(
            histories[i].reshape(1, hist_lens[i]),
            rnn_args = rnn_args,
            temperature = temperature,
            device = device,
        ) for i in range(num_hists)
    ]; probs = [prob_state[0] + eps for prob_state in probs_states]
    states = [prob_state[1] for prob_state in probs_states]

    # Zero out weights on excluded tokens
    for i in range(num_hists):
        probs[i][:,excluded] *= eps/2
        # probs[i] /= probs[i].sum(dim=-1).reshape(-1,1)
        probs[i] /= probs[i].sum(dim=-1)

    # Log probabilities start (num_hists, num_seqs)
    log_probs = [ # (num_seqs, vocab)
        torch.log(probs[i].repeat((num_seqs,1)))
        for i in range(num_hists)
     ]
    widths = [(1,num_seqs,1) for i in range(num_hists)]
    sorted_states = get_initial_sorted_states(states, widths,num_hists)

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
            torch.unsqueeze(seqs[i][:,-1],-1)).cpu()
        for i in range(num_hists)
    ]

    seqs, log_probs = importance_sampling_inner_loop(
        model, seqs, log_probs,
        sorted_states,
        num_hists, seq_lens,
        device, temperature,
        range(1,max(seq_lens),1),
        eps, excluded,
        disable_tqdm = False,
    )

    if batch:
        return torch.stack(seqs,0), torch.stack(log_probs,dim=0), None
    return seqs, log_probs, None

#######################################################################
# Beam search evaluation
#######################################################################


def evaluate_beam_search_lbs(
    model,
    sample_lists,
    excluded_tokens,
    seq_lens,
    sample_prob_lists = None,
    device = "cuda:0",
    excluded = [],
):
    if isinstance(seq_lens, int):
        seq_lens = [seq_lens]*len(sample_lists)
    if sample_prob_lists is None:
        sample_prob_lists = [None]*len(sample_lists)

    lower_bounds = [
        _evaluate_beam_search_lb(
            model,
            sample_lists[i],
            seq_lens[i],
            sample_prob_lists[i],
            device=device,
            excluded=excluded,
        ) for i in range(len(sample_lists))
    ]

    return lower_bounds

def _evaluate_beam_search_lb(
    model,
    samples,
    seq_len,
    sample_probs = None,
    device = "cuda:0",
    excluded = [],
):
    """ Sum up all probabilities from
    beam search to get a lower bound on the
    actual marginal probability
    """
    probs = model.get_next_probs(
        samples,
        rnn_args = None,
        temperature = None,
        device = device,
    )

    assert tuple(probs.shape) == tuple(sample_probs.shape),\
        f"Probabilities are of different shapes: probs: {probs.shape} sampled: {sample_probs.shape}"
    if sample_probs is not None:
        log_probs = torch.log(probs) - sample_probs

    lower_bound = torch.exp(log_probs.sum())
    return lower_bound

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
    ).flatten()[0].item() + 1

    return beam_width


def get_sorted_states_list(
    states,
    repeat_shape,
    num_hists,
):
    """TODO: Docstring for get_initial_sorted_states_list.

    :states: TODO
    :repeat_shape: TODO
    :: TODO
    :returns: TODO

    """
    if isinstance(states[0], tuple):
        sorted_states = [
            (states[i][0].repeat(repeat_shape),
            states[i][1].repeat(repeat_shape)
            ) for i in range(num_hists)
        ]
    else:
        sorted_states = [
            states[i].repeat(repeat_shape)
            for i in range(num_hists)
        ]
    return sorted_states

def get_initial_sorted_states(
    states,
    repeat_shape,
    num_hists,
):
    """TODO: Docstring for get_initial_sorted_states_list.

    :states: TODO
    :repeat_shape: TODO
    :: TODO
    :returns: TODO

    """
    if isinstance(states,list):
        if isinstance(states[0], tuple):
            sorted_states = [
                (states[i][0].repeat(repeat_shape[i]),
                states[i][1].repeat(repeat_shape[i])
                ) for i in range(num_hists)
            ]
        else:
            sorted_states = [
                states[i].repeat(repeat_shape[i])
                for i in range(num_hists)
            ]
    else:
        if isinstance(states, tuple):
            sorted_states = (states[0].repeat(repeat_shape),
                states[1].repeat(repeat_shape))
        else:
            sorted_states = states.repeat(repeat_shape)

    return sorted_states

def get_beam_coverage(
    curr_coverage,
    orig_coverage,
    bw_params,
    index = None,
):
    """TODO: Docstring for get_beam_coverage.

    :curr_coverage: TODO
    :orig_coverage: TODO
    :: TODO
    :returns: TODO

    """

    fixed_width = lambda cov, orig, params,index: orig
    backoff = lambda cov, orig, params,index: cov*orig
    interpolate = lambda cov, orig, params, index: cov - params['interpolate_step'][index]
    assert (bw_params['coverage_type'] in ['fixed_width','backoff','interpolate'])
    roster = {'fixed_width': fixed_width,
              'backoff': backoff,
              'interpolate': interpolate,
              }
    return roster[bw_params['coverage_type']](curr_coverage, orig_coverage, bw_params, index)

def prepare_beams(
    beam_widths,
    num_hists,
    seq_lens,
    bw_params,
):
    roster = {'fixed_width':_prepare_beams_backoff,
              'backoff':_prepare_beams_backoff,
              'interpolate': _prepare_beams_interpolate,
              }
    return roster[bw_params['coverage_type']](
        beam_widths,
        num_hists,seq_lens,
        bw_params
    )

def _prepare_beams_interpolate(
    beam_widths,
    num_hists,
    seq_lens,
    bw_params,
):
    percentile = False
    if isinstance(beam_widths, float):
        assert 0 < beam_widths < 1, "Coverage must be between 0 and 1"
        percentile = True
        orig_beam_widths = [np.power(beam_widths, 1/seq_lens[i]) for i in range(num_hists)]
        bw_params['interpolate_step'] = np.array(
            [(orig_beam_widths[i] - beam_widths)/(seq_lens[i] -1) for i in range(num_hists)]
        )

    elif isinstance(beam_widths, list):
        assert len(beam_widths) == num_hists,"Histories and beam widths not aligned"
        if isinstance(beam_widths[0],float):
            percentile = True
            orig_beam_widths = [np.power(beam_widths[i], 1/seq_lens[i]) for i in range(num_hists)]
            bw_params['interpolate_step'] = np.array(
                [(orig_beam_widths[i] - beam_widths[i])/(seq_lens[i] -1) for i in range(num_hists)]
            )
    else:
        assert False,"Beam data type must be float or List[float]"
    return orig_beam_widths, percentile


def _prepare_beams_backoff(
    beam_widths,
    num_hists,
    seq_lens,
    bw_params,
):
    """TODO: Docstring for prepare_beams.

    :beam_width: TODO
    :returns: TODO

    """
    percentile = False
    if isinstance(beam_widths,int):
        orig_beam_widths = np.array([beam_widths]*num_hists)
    elif isinstance(beam_widths, float):
        percentile = True
        orig_beam_widths = np.array(
            [np.power(beam_widths, 1/seq_lens[i]) for i in range(num_hists)]
        )
    elif isinstance(beam_widths, list):
        assert len(beam_widths) == num_hists,"Histories and beam widths not aligned"
        if isinstance(beam_widths[0],float):
            percentile = True
            orig_beam_widths = np.array(
                [np.pow(beam_widths[i], 1/seq_lens[i]) for i in range(num_hists)]
            )
        else:
            orig_beam_widths = beam_widths
    else:
        assert False,"Beam data type must be int, float or List[int/float]"
    return orig_beam_widths, percentile


def beam_search_inner_loop(
    model, seqs,
    sorted_probs_inds,
    sorted_states,
    all_beam_coverages, orig_beam_coverage,
    all_beam_widths,
    percentile,
    num_hists, seq_lens,
    device, temperature,
    iter_range, eps,
    excluded, vocab_size,
    disable_tqdm = False,
    bw_params = None,
):
    """TODO: Docstring for beam_search_inner_loop.

    :iter_range: TODO
    :sorted_states: TODO
    :seq_lens: TODO
    :device: TODO
    :temperature: TODO
    :disable_tqdm: TODO
    :: TODO
    :returns: TODO

    """

    # Iterate through all sequences
    for pos in tqdm(iter_range,disable = disable_tqdm):

        # Get new probabilities
        # List[(beam_width, vocab)] len num_hists
        new_probs_states = [
                None if (pos >= seq_lens[i]) else
                model.get_next_probs(
                    seqs[i][...,-1].reshape(-1,1),
                    rnn_args = sorted_states[i],
                    temperature = temperature,
                    device = device,
                ) for i in range(num_hists)
        ]; new_probs,new_states = list(zip(*[
            (new_prob_state[0] + eps, new_prob_state[1])
            if new_prob_state is not None else (None,None)
            for new_prob_state in new_probs_states]))
        new_probs, new_states = list(new_probs), list(new_states)

        # Knock down excluded probabilities
        for i in range(num_hists):
            if new_probs[i] is not None:
                new_probs[i][:,excluded] = eps/2
                new_probs[i] /= new_probs[i].sum(dim=-1).reshape(-1,1)

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
                    all_beam_coverages[i][-1], sorted_probs,
                    percentile=percentile,
                ); all_beam_widths[i].append(beam_width)
                if percentile:
                    new_coverage = get_beam_coverage(all_beam_coverages[i][-1],
                                                     orig_beam_coverage[i],
                                                     bw_params, index = i)
                    # print(new_coverage)
                    all_beam_coverages[i].append(new_coverage)

                # Match sequences with their next best token
                tokens = sorted_probs_inds[i][1][:beam_width].cpu()
                seq_inds = tokens // vocab_size
                best_tokens = tokens % vocab_size

                # (beam_width, curr_seq_len+1)
                # Need to expand this if the size is growing
                sorted_probs_inds[i] = (
                    sorted_probs_inds[i][0][seq_inds],
                    best_tokens
                )
                # Check for multiple hidden states or not
                if isinstance(new_states[i], tuple):
                    sorted_states[i] = (new_states[i][0][:,seq_inds,:],
                                        new_states[i][1][:,seq_inds,:])
                else: sorted_states[i] = new_states[i][:,seq_inds,:]

                seqs[i] = \
                    torch.cat(
                        (
                            # (beam_width, curr_seq_len)
                            seqs[i][seq_inds, :],
                            # (beam_width,1)
                            best_tokens.reshape(-1,1),
                        ), dim = 1
                    )
    return seqs, probs


#######################################################################
# Beam Search (batch and variable are a single method)
#######################################################################

@torch.no_grad()
def sample_beam_search(
    histories,
    vocab_size,
    beam_widths,
    total_seq_lens,
    model = None,
    temperature = 1,
    rnn_args = None,
    excluded = [],
    disable_tqdm = False,
    device = 'cpu',
    eps = 1e-10,
    bw_params = {'coverage_type':'backoff'},
    return_beams = True,
    **kwargs,
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
    if isinstance(total_seq_lens, int): total_seq_lens = [total_seq_lens]*num_hists
    seq_lens = [total_seq_len_i - hist_len for total_seq_len_i,hist_len
                in zip(total_seq_lens,hist_lens)]
    orig_beams, percentile = prepare_beams(beam_widths,num_hists,seq_lens, bw_params)
    all_beam_coverages = [[orig_beam] for orig_beam in orig_beams]
    excluded = set(excluded)
    legal_vocab = np.array(
        [val for val in np.arange(vocab_size) if val not in excluded]
    ); excluded = list(excluded)

    # Get probabilities from model for each history
    # (num_hists, vocab)
    probs_states = [
        model.get_next_probs(
            histories[i].reshape(1, hist_lens[i]),
            rnn_args = rnn_args,
            temperature = temperature,
            device = device,
        ) for i in range(num_hists)
    ]; probs = [prob_state[0] + eps for prob_state in probs_states]
    states = [prob_state[1] for prob_state in probs_states]

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
    ]
    # Initial seqs
    # List[(beam_width, hist_len + 1)] len num_hists
    beam_widths = [
        get_beam_width(
            beam_coverage[-1], prob,
            percentile=percentile,
        ) for beam_coverage,prob in
        zip(all_beam_coverages, probs)
    ]; all_beam_widths = [[bw] for bw in beam_widths]

    # Create the logarithm here to seed sum
    sorted_probs_inds = [
        (torch.log(sorted_prob), sorted_ind)
        for sorted_prob, sorted_ind in sorted_probs_inds
    ]

    # Initial seqs
    # List[(beam_width, hist_len + 1)] len num_hists
    widths = [(1,min(bw, vocab_size),1) for bw in beam_widths]
    sorted_states = get_initial_sorted_states(states, widths,num_hists)

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

    seqs, probs = beam_search_inner_loop(
        model, seqs,
        sorted_probs_inds,
        sorted_states,
        all_beam_coverages, orig_beams,
        all_beam_widths,
        percentile,
        num_hists, seq_lens,
        device, temperature,
        range(1,max(seq_lens) + 1,1),
        eps, excluded, vocab_size,
        disable_tqdm = disable_tqdm,
        bw_params = bw_params
    )

    if bw_params['coverage_type'] == "fixed_width" and not percentile:
        seqs, probs =  torch.stack(seqs, dim = 0), torch.stack(probs,dim =0)
    if return_beams:
        return seqs, probs, (all_beam_widths,all_beam_coverages)
    return seqs, probs, None


#################################################################################
#   TESTING Orchestration
#################################################################################

def sample(
    dataloader,
    args,
    model = None,
    **kwargs,
):
    """Sample from any of these methods given an
    input dataloader, arguments, and potentially a model

    :dataloader: TODO
    :args: TODO
    :model: TODO
    :: TODO
    :returns: TODO

    """
    roster = {
        "beam_search": sample_beam_search,
        "mc_random": (mc_sample_random_batch if
        args.sample_args['coverage_type'] == "fixed_width"
                      else mc_sample_random_list),
        "mc_importance": mc_sample_importance,
    };
    sampler = roster[args.sample_type]
    args.model = model;
    output = {"settings":vars(args)}

    all_seqs = []; all_probs = []; all_beam = []; all_covs = []
    for dbatch in dataloader:
        batched = False
        if isinstance(args.hist_len,int):
            args.hist_len = [args.hist_len]*dbatch.shape[0]
            batched = True
        data_batch =[dbatch[i,:args.hist_len[i]] for i in range(dbatch.shape[0])]
        if batched: data_batch = torch.stack(data_batch, dim = 0).cpu()
        kwargs = vars(args)
        # del kwargs['vocab_size']
        seqs, probs, beams_covs = sampler(data_batch,**kwargs)

        #Tensors
        # print(probs[0].shape)
        seqs = [seq.numpy() for seq in seqs] if isinstance(seqs,list) else [seqs.numpy()]
        probs = ([None] if probs is not None
                 else ([prob.numpy() for prob in probs] if isinstance(seqs, list)
                 else ([probs.numpy()])))
        all_seqs += seqs
        all_probs += probs
        print(len(seqs))
        sys.exit(1)

        # Lists
        if beams_covs is not None:
            all_beams += beams_covs[0]
            all_covs += beams_covs[1]

    output['beams'] = all_beams
    output['probabilities'] = all_probs
    output['sequences'] = all_seqs
    output['covs'] = all_covs
    args.model = None

    return output









