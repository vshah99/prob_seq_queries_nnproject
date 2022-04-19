#################################################################################
#
#             Project Title:  Beam Search Sampling Tree
#             Author:         Sam Showalter
#             Date:           2022-04-19
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
from collections import defaultdict

#################################################################################
#   Function-Class Declaration
#################################################################################

class BSNode(object):

    """Beam Search sample node"""

    def __init__(
        self,
        symbol,
        parent,
        marginals,
        depth,
    ):
        """TODO: to be defined.

        :: TODO

        """
        self.symbol = symbol
        self.parent = parent
        self.marginals = marginals
        self.depth = depth
        self.children = defaultdict(dict)

class BeamSearchSampleTree(object):

    """
    Tree data structure for sampling leftover (non-searched)
    sequences to estimate lost probability coverage
    """

    def __init__(
        self,
        text_dict,
    ):
        """TODO: to be defined.

        :: TODO

        """
        self.id_to_char = text_dict['id_to_char']
        self.char_to_id = text_dict['char_to_id']
        self.vocab_size = len(self.char_to_id)
        self.BOS = self.char_to_id['<BOS>']
        self.depth_sizes = [1]
        self.depth_dict = defaultdict(dict)
        self.rooted = False

    def add_root_node(
        self,
        marginals,
    ):
        self.root = BSNode(self.BOS,None,marginals,depth=0)
        self._add_depth(0,self.root)

    def add_child_nodes(
        self,
        symbols,
        parent_ids,
        marginals,
        depth,
    ):
        for s,pid,marg in zip(symbols,parent_ids,marginals):
            self._add_child_node(s,pid,marg,depth)

    def _add_child_node(
        self,
        symbol,
        parent_id,
        marginals,
        depth,
    ):
        parent = self.depth_dict[depth-1][pid]
        child_node = BSNode(symbol,parent,marginals,depth)

        parent.children[symbol] = child_node
        self._add_depth(depth, node)

    def _add_depth(
        self,
        depth,
        node,
    ):
        if len(self.depth_sizes) >= depth:
            self.depth_sizes.append(0)
        self.depth_sizes += 1
        self.depth_dict[depth][(node.parent.symbol, node.symbol)]=node

    def prune(
        self,
    ):
        pass


    @torch.no_grad()
    def run_beam_search_lb(self,
        hist, num_beams, sample_len, model, excluded_terms, interp_func,
        batch_size, device, vocab_size, **kwargs):
        assert(isinstance(num_beams, (int, float)))
        assert(len(hist.shape) == 1)

        beams, rnn_args = hist.unsqueeze(0), None  # beams only represents what needs to be processed by the model in the next step
        sequences = None
        cur_log_probs = torch.zeros((1,), dtype=torch.float32)  # (num of current beams,)
        cur_restricted_log_probs = cur_log_probs.clone()  # sum of restricted probabilities
        num_beams_over_time = []
        for n_cur in range(sample_len):
            logits, states = model.get_next_probs(beams, rnn_args=rnn_args, return_logits = True,
                                                max_batch_size=batch_size,device=device)
            next_log_probs = torch.log_softmax(logits, dim=-1)  # (num of current beams, vocab_size)
            # We need this for each symbol, (tracked based on beams, could use sequence id)
            next_log_probs[..., excluded_terms] = -float('inf')
            next_restricted_log_probs = torch.log_softmax(next_log_probs, dim=-1)
            if n_cur == 0: # Add root node
                self.add_root_node(next_restricted_log_probs)
            else:
                self.add_child_nodes(symbols,curr_parents,
                            next_restricted_log_probs,
                            depth=n_cur)

            next_log_probs = cur_log_probs.unsqueeze(-1) + next_log_probs
            next_log_probs = next_log_probs.view(-1)
            next_restricted_log_probs = cur_restricted_log_probs.unsqueeze(-1) + next_restricted_log_probs
            next_restricted_log_probs = next_restricted_log_probs.view(-1)

            if isinstance(num_beams, int):
                next_restricted_log_probs = top_k_top_p_filtering(next_restricted_log_probs, top_k=num_beams, is_log_prob=True)
            else:  # isinstance(num_beams, float)
                num_beams_cur = interp_func(num_beams, n_cur, sample_len)
                next_restricted_log_probs = top_k_top_p_filtering(next_restricted_log_probs, top_p=num_beams_cur, is_log_prob=True)

            next_log_probs = next_log_probs.masked_fill(next_restricted_log_probs == -float('inf'), -float('inf'))
            indices = torch.arange(0, next_log_probs.shape[0], device=beams.device)[next_log_probs != -float('inf')]
            # Sequence indices we will need for next piece
            seq_inds = torch.div(indices, vocab_size, rounding_mode='trunc')  # equivalent to: indices // args.vocab_size
            # (beams x 1)
            beams = (indices % vocab_size).unsqueeze(-1)
            if sequences is None:
                sequences = beams.clone()
            else:
                sequences =
            cur_log_probs = next_log_probs[indices]
            cur_restricted_log_probs = next_restricted_log_probs[indices]
            rnn_args = states
            if isinstance(rnn_args, tuple):
                rnn_args = rnn_args[0][..., seq_inds, :], rnn_args[1][..., seq_inds, :]
            else:
                rnn_args = rnn_args[..., seq_inds, :]

            # print(n_cur, cur_log_probs.shape[0])
            num_beams_over_time.append(cur_log_probs.shape[0])

        logits, states = model.get_next_probs(beams, rnn_args=rnn_args, device=device, return_logits=True,
                                            max_batch_size=batch_size)
        next_log_probs = cur_log_probs.unsqueeze(-1) + torch.log_softmax(logits, dim=-1)
        return {
            "dist_lower_bound": next_log_probs.exp().sum(dim=0).cpu(),
            "true_coverage": cur_log_probs.exp().sum().cpu(),
            "restricted_coverage": cur_restricted_log_probs.exp().sum().cpu(),
            "num_beams": num_beams_over_time,
        }


#######################################################################
# Sampling orchestration function
#######################################################################



#################################################################################
#   Main Method
#################################################################################



