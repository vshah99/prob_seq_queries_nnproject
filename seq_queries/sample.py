#################################################################################
#
#             Project Title:  Sampling
#             Author:         Sam Showalter and Alex Boyd
#             Date:           2022.04.13
#
#################################################################################

#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy
from collections import defaultdict
import pickle as pkl

import numpy as np
import random
import torch
import torch.nn as nn

from tqdm import tqdm
# from .data import load_text, process_data
from .model import CausalLM, MaskedLM
from .utils import top_k_top_p_filtering, min_variance_top_k
from .tree import BeamSearchSampleTree

#################################################################################
#   Function-Class Declaration
#################################################################################

def uniform_proposal(hists, seq_len, model, vocab_size, excluded_terms,
                     batch_size, device='cpu', **kwargs):
    assert(len(hists.shape) == 2)

    # Uniformly sample across the restricted vocabulary indices
    samples = torch.randint(low=0, high=vocab_size-len(excluded_terms), size=(hists.shape[0], seq_len), device=hists.device)
    for item in sorted(excluded_terms):
        samples[samples>=item] += 1
    assert(samples.max() < vocab_size)

    logits,hidden_states = model.get_next_probs(torch.cat((hists, samples), dim=-1), return_forward_only=True,
                                                device=device, return_logits=True, max_batch_size=batch_size)  #, device=device)
    model.model_iters -= hists.shape[0]*hists.shape[1]
    model_log_prob = torch.log_softmax(logits, dim=-1)[..., -(seq_len+1):-1, :]
    model_log_prob = torch.gather(model_log_prob, dim=-1, index=samples.unsqueeze(-1)).squeeze(-1).sum(dim=-1)  # grab specific log probabilities

    return {
        "proposal_log_prob": -seq_len * np.log(vocab_size - len(excluded_terms)),
        "model_log_prob": model_log_prob.unsqueeze(-1),
        "samples": samples,
        "logits": logits,
        "next_log_dist": torch.log_softmax(logits, dim=-1)[..., -1, :],
    }

def lm_proposal(hists, seq_len, model, vocab_size, excluded_terms,
                batch_size=128,device='cpu',top_k=0, top_p=1.0, temperature=1.0,  **kwargs):
    assert(len(hists.shape) == 2)

    proposal_log_prob, model_log_prob = 0.0, 0.0
    samples = []; all_logits = []; started = False
    last_sample, rnn_args = hists, None
    for _ in range(seq_len):
        logits, rnn_args = model.get_next_probs(last_sample, rnn_args=rnn_args, max_batch_size=batch_size,
                                                device=device, return_logits=True)
        if not started: model.model_iters = 0; started= True
        all_logits.append(logits)

        proposal_logits = logits.clone()
        proposal_logits[..., excluded_terms] = -float('inf')
        proposal_logits = torch.log_softmax(top_k_top_p_filtering(proposal_logits/temperature, top_k=top_k, top_p=top_p), dim=-1)

        last_sample = torch.distributions.Categorical(logits=proposal_logits).sample().unsqueeze(-1)
        proposal_log_prob += torch.gather(proposal_logits, dim=-1, index=last_sample).squeeze(-1)
        model_log_prob += torch.gather(torch.log_softmax(logits, dim=-1), dim=-1, index=last_sample).squeeze(-1)
        samples.append(last_sample)

    logits, _ = model.get_next_probs(last_sample, rnn_args=rnn_args, device=device,
                                     max_batch_size=batch_size,return_logits=True)  # get last subsequent distribution
    all_logits.append(logits)

    samples = torch.cat((samples + [torch.ones_like(samples[0])*excluded_terms[0]]), dim=-1)
    return {
        "proposal_log_prob": proposal_log_prob.unsqueeze(-1),
        "model_log_prob": model_log_prob.unsqueeze(-1),
        "samples": samples,
        "logits": torch.stack(all_logits,dim=1),
        "next_log_dist": torch.log_softmax(logits, dim=-1),
    }

@torch.no_grad()
def mc_estimate(hist, num_mc_samples, seq_len, model, excluded_terms, proposal_func,
                vocab_size, batch_size=128,temperature=1, top_k=0, top_p=0.0, device='cpu',
                cat_list = ['sample_estimates'],
                sub_estimates=None,**kwargs):
    model.model_iters = 0
    assert(len(hist.shape) == 1)  # (hist_seq_len), Only conditions on a single history
    # assert(len(excluded_terms) == 1) # For most experiments
    out_dict = defaultdict(list)
    remaining_samples = num_mc_samples
    while remaining_samples > 0:
        sample_out = proposal_func(
            hists=hist.unsqueeze(0).expand(min(remaining_samples, batch_size), -1),
            seq_len=seq_len,
            model=model,
            vocab_size=vocab_size,
            excluded_terms=excluded_terms,
            top_k=top_k,
            top_p=top_p,
            device=device,
            batch_size=batch_size,
            temperature=temperature,
        )
        remaining_samples -= batch_size
        term_log_prob = sample_out["next_log_dist"] + sample_out["model_log_prob"] - sample_out["proposal_log_prob"]

        out_dict['sample_estimates'].append(term_log_prob.exp().cpu())
        # out_dict['q_log_prob'].append(sample_out['proposal_log_prob'])

    for item in cat_list:
        out_dict[item] = torch.cat(out_dict[item],dim=0)

    if sub_estimates:
        # (samples x vocab) -> (sub-estimates x vocab)
        out_dict['sample_estimates'] = torch.stack(
            # (vocab)
            [out_dict['sample_estimates'][:s].mean(dim=0).flatten()
             for s in sorted(sub_estimates)
        ]).squeeze()
        out_dict['sample_estimate_var'] = torch.stack(
            # (vocab)
            [out_dict['sample_estimates'][:s].var(dim=0).flatten()
             for s in sorted(sub_estimates)
        ]).squeeze()
        out_dict['model_iters'] = torch.LongTensor(
            [sub_est * seq_len for sub_est in sub_estimates]
        )

    else:
        out_dict['model_iters'] = torch.LongTensor([model.model_iters])
        out_dict['sample_estimate_var'] =torch.var(out_dict['sample_estimates'],dim=0)
        out_dict['sample_estimate_mean'] =out_dict['sample_estimates'].mean(dim=0)
    return out_dict

@torch.no_grad()
def beam_search_is_hybrid(hist, num_beams,num_mc_samples, seq_len, model, excluded_terms, interp_func,
                          batch_size, device, vocab_size,
                          beam_search_outputs=['num_beams','true_coverage','restricted_coverage'],
                          min_variance=False,min_var_reduction=0.0,
                          text_dict=None, **kwargs):
    model.model_iters = 0
    beam_search_output =beam_search_lower_bound(
        hist, num_beams, seq_len, model, excluded_terms, interp_func,
        batch_size, device, vocab_size, bs_tree=BeamSearchSampleTree(text_dict),
        min_variance=min_variance,min_var_reduction=min_var_reduction, **kwargs)
    tree = beam_search_output['tree']
    tree.prune()

    hybrid_estimate = tree_is_estimate(
        tree,
        beam_search_output['bs_lower_bound'],
        num_mc_samples, seq_len, model,
        excluded_terms, batch_size, device,
        **kwargs
    )
    for bso in beam_search_outputs:
        hybrid_estimate[bso] = beam_search_output[bso]

    return hybrid_estimate



@torch.no_grad()
def tree_is_estimate(
    tree,
    bs_lower_bound,
    num_mc_samples,
    seq_len,
    model,
    excluded_terms,
    batch_size,
    device,
    sub_estimates=None,
    **kwargs,
):
    # Sample each sequence individually from tree
    log_p_totals, log_q_totals = [], []
    hidden_states, num_remaining_steps = [], []
    last_tokens = []

    for _ in range(num_mc_samples):
        log_p, log_q, hs, depth_reached, last_token, _ = tree.sample_sequence(seq_len)
        log_p_totals.append(log_p)
        log_q_totals.append(log_q)
        hidden_states.append(hs)
        num_remaining_steps.append(seq_len - depth_reached)
        last_tokens.append(last_token)

    log_p_totals = torch.stack(log_p_totals, dim=0)
    log_q_totals = torch.stack(log_q_totals, dim=0)
    if isinstance(hs, tuple):
        hidden_states = (
            torch.stack([h[0] for h in hidden_states], dim=1),
            torch.stack([h[1] for h in hidden_states], dim=1),
        )
    else:
        hidden_states = torch.stack(hidden_states, dim=1)
    num_remaining_steps = torch.tensor(num_remaining_steps, dtype=torch.int32, device=log_p_totals.device)
    last_tokens = torch.stack(last_tokens, dim=0).unsqueeze(1)  # need to have a sequence length of 1

    # Finish sampling incomplete sequences from model
    model_iters = [model.model_iters + num_remaining_steps.sum()]

    if sub_estimates:
        model_iters = []
        samples_per_effort = [
            (num_remaining_steps == i).sum().item() for i in range(num_remaining_steps.max()+1)
        ];
        j = 0
        for i in range(len(sub_estimates)):
            total_samp = 0; total_cost = model.model_iters
            curr_model_iters = model.model_iters
            for j in range(len(samples_per_effort)):
                if total_samp < sub_estimates[i]:
                    samp_left = min(sub_estimates[i] - total_samp,
                                    samples_per_effort[j])
                    total_samp += samples_per_effort[j]
                    # print(total_samp,sub_estimates[i])
                    total_cost += j*samp_left
                if (total_samp >= sub_estimates[i]):
                    model_iters.append(total_cost)
                    break

    while (num_remaining_steps > 0).any():
        to_update = num_remaining_steps > 0
        if isinstance(hidden_states, tuple):
            rnn_args = (hidden_states[0][..., to_update, :], hidden_states[1][..., to_update, :])
        else:
            rnn_args = hidden_states[..., to_update, :]
        logits, rnn_args = model.get_next_probs(
            last_tokens[to_update, :],
            rnn_args=rnn_args,
            max_batch_size=batch_size,
            device=device,
            return_logits=True,
        )

        proposal_logits = logits.clone()
        proposal_logits[..., excluded_terms] = -float('inf')
        logits, proposal_logits = torch.log_softmax(logits, dim=-1), torch.log_softmax(proposal_logits, dim=-1)
        last_sample = torch.distributions.Categorical(logits=proposal_logits).sample().unsqueeze(-1)
        log_q_totals[to_update] += torch.gather(proposal_logits, dim=-1, index=last_sample).squeeze(-1)
        log_p_totals[to_update] += torch.gather(logits, dim=-1, index=last_sample).squeeze(-1)
        if isinstance(hidden_states, tuple):
            hidden_states[0][..., to_update, :] = rnn_args[0]
            hidden_states[1][..., to_update, :] = rnn_args[1]
        else:
            hidden_states[..., to_update, :] = rnn_args
        num_remaining_steps[to_update] -= 1
        last_tokens[to_update, :] = last_sample

    # Compute final distributions for estimate
    next_log_dist, _ = model.get_next_probs(
        last_tokens,
        hidden_states,
        max_batch_size=batch_size,
        device=device,
        return_logits=True,
    )
    next_log_dist = torch.log_softmax(next_log_dist, dim=-1)  # (num_seqs, vocab_size)
    dist_estimate = next_log_dist + log_p_totals.unsqueeze(dim=-1) - log_q_totals.unsqueeze(dim=-1)
    dist_estimate = dist_estimate.exp().cpu()
    dist_est_var = dist_estimate.var(dim=0)
    model_iters = [model_iter + sub_est for model_iter, sub_est in zip(model_iters,sub_estimates)]

    if sub_estimates is not None and len(sub_estimates) > 0:
        # (samples x vocab) -> (sub-estimates x vocab)
        dist_estimate = torch.stack(
            # (vocab)
            [dist_estimate[:s].mean(dim=0).flatten()
             for s in sorted(sub_estimates)
        ]).squeeze()
        dist_est_var = torch.stack(
            # (vocab)
            [dist_estimate[:s].var(dim=0).flatten()
             for s in sorted(sub_estimates)
        ]).squeeze()

    return {
        'bs_lower_bound':bs_lower_bound,
        'is_estimates':dist_estimate,
        'sample_estimates': bs_lower_bound + dist_estimate,
        'sample_estimate_var': dist_est_var,
        'sample_estimate_mean':(bs_lower_bound + dist_estimate).mean(dim=0) if not sub_estimates else torch.Tensor([]),
        'model_iters': torch.LongTensor(model_iters),
    }

def geom_interp(target_pct, n_current, n_end):
    return target_pct ** ((n_current + 1) / n_end)  # add 1 due to 0-indexing

def lin_interp(target_pct, n_current, n_end):
    a = geom_interp(target_pct=target_pct, n_current=0, n_end=n_end)
    b = target_pct
    t = n_current / (n_end - 1)
    return a * (1 - t) + b * t

@torch.no_grad()
def beam_search_lower_bound(hist, num_beams, seq_len, model, excluded_terms,
                            interp_func, batch_size, device, vocab_size,
                            bs_tree=None, store_intermediate_lbs=False, sub_estimates=None,
                            min_variance=False,min_var_reduction=0.0, **kwargs):
    assert(isinstance(num_beams, (int, float)))
    assert(len(hist.shape) == 1)

    model.model_iters = 0; started = False; intermediate_lbs = []
    beams, rnn_args = hist.unsqueeze(0), None  # beams only represents what needs to be processed by the model in the next step
    cur_log_probs = torch.zeros((1,), dtype=torch.float32)  # (num of current beams,)
    cur_restricted_log_probs = cur_log_probs.clone()  # sum of restricted probabilities
    num_beams_over_time = []
    for n_cur in range(seq_len):
        logits, states = model.get_next_probs(beams, rnn_args=rnn_args, return_logits = True,
                                            max_batch_size=batch_size,device=device)
        if not started: model.model_iters = 0; started= True
        next_log_probs = torch.log_softmax(logits, dim=-1)  # (num of current beams, vocab_size)
        stored_next_log_probs = next_log_probs.clone()
        # We need this for each symbol, (tracked based on beams, could use sequence id)
        next_log_probs[..., excluded_terms] = -float('inf')
        next_restricted_log_probs = torch.log_softmax(next_log_probs, dim=-1)
        stored_restricted_log_probs = next_restricted_log_probs.clone()

        next_log_probs = cur_log_probs.unsqueeze(-1) + next_log_probs

        # If we need to store intermediate results
        if store_intermediate_lbs:
            intermediate_lbs.append(next_log_probs.exp().sum(dim=0).cpu())

        next_log_probs = next_log_probs.view(-1)
        next_restricted_log_probs = cur_restricted_log_probs.unsqueeze(-1) + next_restricted_log_probs
        next_restricted_log_probs = next_restricted_log_probs.view(-1)

        if min_variance:
                next_restricted_log_probs = min_variance_top_k(next_restricted_log_probs, min_var_reduction=min_var_reduction, is_log_prob=True)
        elif isinstance(num_beams, int):
                next_restricted_log_probs = top_k_top_p_filtering(next_restricted_log_probs, top_k=num_beams, is_log_prob=True)
        else:  # isinstance(num_beams, float)
            num_beams_cur = interp_func(num_beams, n_cur, seq_len)
            next_restricted_log_probs = top_k_top_p_filtering(next_restricted_log_probs, top_p=num_beams_cur, is_log_prob=True)

        next_log_probs = next_log_probs.masked_fill(next_restricted_log_probs == -float('inf'), -float('inf'))
        if bs_tree is not None:
            if n_cur == 0: # Add root node
                parents = bs_tree.add_root_node(
                    log_q_conditionals=stored_restricted_log_probs,
                    log_p_conditionals=stored_next_log_probs,
                    hidden_state=states,
                )
            else:
                parents = bs_tree.add_child_nodes(
                    symbols=beams,
                    parents=parents,
                    log_q_conditionals=stored_restricted_log_probs,
                    log_p_conditionals=stored_next_log_probs,
                    hidden_states=states,
                    parent_ids=seq_inds,
                    depth=n_cur,
                )

        # (beams x 1)
        indices = torch.arange(0, next_log_probs.shape[0], device=beams.device)[next_log_probs != -float('inf')]
        if n_cur ==0 and bs_tree is not None: parents = parents * indices.shape[0]
        # Sequence indices we will need for next piece
        seq_inds = torch.div(indices, vocab_size, rounding_mode='trunc')  # equivalent to: indices // args.vocab_size
        beams = (indices % vocab_size).unsqueeze(-1)
        cur_log_probs = next_log_probs[indices]
        cur_restricted_log_probs = next_restricted_log_probs[indices]
        rnn_args = states
        if isinstance(rnn_args, tuple):
            rnn_args = rnn_args[0][..., seq_inds, :], rnn_args[1][..., seq_inds, :]
        else:
            rnn_args = rnn_args[..., seq_inds, :]

        num_beams_over_time.append(cur_log_probs.shape[0])

    logits, states = model.get_next_probs(beams, rnn_args=rnn_args, device=device, return_logits=True,
                                        max_batch_size=batch_size)
    final_log_probs = torch.log_softmax(logits,dim=-1)
    next_log_probs = cur_log_probs.unsqueeze(-1) + final_log_probs
    if bs_tree is not None:
        bs_tree.add_child_nodes(
                    beams,parents,
                    final_log_probs,
                    final_log_probs.clone(), #next_log_probs,
                    states, seq_inds,
                    depth=n_cur+1)

    out_dict = {
        "tree": bs_tree,
        "bs_lower_bound": next_log_probs.exp(),
        "true_coverage": cur_log_probs.exp(),
        "restricted_coverage": cur_restricted_log_probs.exp(),
        "num_beams": torch.LongTensor(num_beams_over_time),
        "model_iters": torch.LongTensor([model.model_iters]),
        "intermediate_lbs": (torch.Tensor([]) if not store_intermediate_lbs
                             else torch.stack(intermediate_lbs)),
    }

    to_accumulate = ['bs_lower_bound','true_coverage',
                    'restricted_coverage']
        # (samples x vocab) -> (sub-estimates x vocab)
    if sub_estimates:
        per_step_iters = np.array([
            [min(sub_est,(vocab_size - len(excluded_terms))**(i+1))
                            for i in range(seq_len)]
                for sub_est in sub_estimates])
        out_dict['model_iters'] = torch.LongTensor(per_step_iters.sum(axis=1))
        out_dict['num_beams'] = torch.LongTensor(sub_estimates)
        for term in to_accumulate:
            out_dict[term] = torch.stack(
                # (vocab)
                [out_dict[term][:s].sum(dim=0).flatten()
                for s in sorted(sub_estimates)
            ]).squeeze().cpu()
    else:
        for term in to_accumulate:
            out_dict[term] = out_dict[term].sum(dim=0)

    return out_dict



#######################################################################
# Sampling orchestration function
#######################################################################

@torch.no_grad()
def sample(
    args,
    dataloader,
    model = None,
    **kwargs,):
    """Sample from any of these methods given an
    input dataloader, arguments, and potentially a model
    :dataloader: TODO
    :args: TODO
    :model: TODO
    :: TODO
    :returns: TODO
    """
    args.model = model; print();
    output = {"sample_estimates":[]}

    def _tensor_output(key, data,output=output):
        if key not in output: output[key] = []
        output[key].append(torch.Tensor([db[key] for db in data]))

    def _stack_output(key, data,output=output):
        if key not in output: output[key] = []
        output[key].append(torch.stack([
            torch.Tensor(db[key]) for db in data]))

    def _consolidate_output(key,output=output):
        output[key] = torch.cat(output[key])

    for dbatch in tqdm(dataloader, disable=args.disable_tqdm):
        data_list = []
        data_batch =[dbatch[i,:args.hist_len] for i in range(dbatch.shape[0])]

        for i in range(dbatch.shape[0]):
            if i%10 == 0 and args.disable_tqdm:
                print(".",end="",flush=True)
            sample = data_batch[i]
            args.seq_lejn= args.total_seq_len - args.hist_len
            kwargs = vars(args)
            # print(''.join([args.text_dict['id_to_char'][s] for s in sample.tolist()]))
            data_list.append(args.estimate_type(sample,**kwargs))


        if "beam_search" in args.estimate_type.__name__:
            _stack_output('num_beams',data_list)
            _stack_output('dist_lower_bound',data_list)
            _tensor_output('true_coverage',data_list)
            _tensor_output('restricted_coverage',data_list)
        else:
            output['sample_estimates'] += data_list

    if "beam_search" in args.estimate_type.__name__:
        _consolidate_output("num_beams")
        _consolidate_output("true_coverage")
        _consolidate_output("restricted_coverage")
        _consolidate_output("dist_lower_bound")
    else:
        output['sample_estimates'] =torch.stack(output['sample_estimates'],
                                                dim=0)

    args.model = None
    output['metadata'] = vars(args)
    return output

#################################################################################
#   Main Method
#################################################################################



