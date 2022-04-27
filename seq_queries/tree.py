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

from .utils import top_k_top_p_filtering, _tup_cpu, _hidden_state_select

#################################################################################
#   Function-Class Declaration
#################################################################################

class BSNode(object):

    """Beam Search sample node"""

    def __init__(
        self,
        symbol,
        parent,
        log_q_conditionals,
        log_p_conditionals,
        hidden_state,
        tree,
        depth,
    ):
        """TODO: to be defined.

        :: TODO

        """
        self.symbol = symbol
        self.parent = parent
        self.depth = depth
        self.tree = tree
        # Send in marginals, don't transform here
        self.q_conditionals = log_q_conditionals.exp().flatten().cpu()
        self.p_conditionals = log_p_conditionals.exp().flatten().cpu()
        self.hidden_state = _tup_cpu(hidden_state, force=False)
        self.children = defaultdict(dict)
        self.total_mass = 1.0

    @property
    def lineage(self):
        lineage = [self.symbol]
        tracker = self.parent
        while tracker is not None:
            lineage.append(tracker.symbol)
            tracker = tracker.parent
        return list(reversed(lineage))

    def __str__(self):
        return "Node({})".format(
            "".join([self.tree.id_to_char[s] for s in self.lineage]))

    def __unicode__(self):
        return u"Node({})".format(
            "".join([self.tree.id_to_char[s] for s in self.lineage]))

    def __repr__(self):
        return "Node({})".format(
            "".join([self.tree.id_to_char[s] for s in self.lineage]))

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
        self.depth_sizes = [0]
        self.depth_dict = defaultdict(list)

    def add_root_node(
        self,
        log_q_conditionals,
        log_p_conditionals,
        hidden_state,
    ):
        self.root = BSNode(self.BOS,None,
                           log_q_conditionals,
                           log_p_conditionals,
                           _hidden_state_select(hidden_state,0),
                           tree=self,
                           depth=0)
        self._add_depth(0,self.root)
        return [self.root]

    def add_child_nodes(
        self, symbols, parents,
        log_q_conditionals,
        log_p_conditionals,
        hidden_states,
        parent_ids,depth,
    ):
        #(beams x vocab)
        new_parents = []
        # Not right, should not be parents
        assert torch.max(parent_ids) < len(parents),\
            "Parent list of length {}, got index {}".format(len(parents),torch.max(parent_ids))
        assert parent_ids.shape[0] == symbols.shape[0] == log_q_conditionals.shape[0] == log_p_conditionals.shape[0],\
            "Parent ids were of shape {} but symbols were of shape {} and q was of shape {} and p was of shape {} and h_state was of shape {}"\
            .format(parent_ids.shape,symbols.shape, log_q_conditionals.shape, log_p_conditionals.shape,hidden_states[0].shape)
        
        for i in range(symbols.shape[0]):
            s,pid,q,p,h = (symbols[i],parent_ids[i],log_q_conditionals[i],
                            log_p_conditionals[i],_hidden_state_select(hidden_states,i))
            new_parents.append(self._add_child_node(s.item(),parents[pid],q,p,h,depth))

        return new_parents

    def _add_child_node(
        self, symbol, parent,
        log_q_conditional,
        log_p_conditional,
        hidden_state, depth,
    ):
        assert 0 <= symbol <= self.vocab_size,\
            f"Invalid symbol: {symbol}, vocab size = {self.vocab_size}"
        child_node = BSNode(symbol,parent,
                            log_q_conditional,
                            log_p_conditional,
                            hidden_state,self,depth)
        parent.children[symbol] = child_node
        self._add_depth(depth, child_node)
        return child_node

    def _add_depth(
        self,
        depth,
        node,
    ):
        if len(self.depth_sizes) <= depth:
            self.depth_sizes.append(0)
        self.depth_sizes[depth] += 1
        # Probably don't need this
        self.depth_dict[depth].append(node)

    def _respect_bs_support(
        self,
        node
    ):
        child_symbols = torch.LongTensor([c.symbol for c in node.children.values()])
        node.q_conditionals[child_symbols] *= 0
        node.total_mass = node.q_conditionals.sum()
        node.q_conditionals = node.q_conditionals / node.total_mass

    def _adjust_marginal_probabilities_by_node(self,node):
        child_symbols = torch.LongTensor([c.symbol for c in node.children.values()])
        child_mass = torch.Tensor([c.total_mass for c in node.children.values()])
        adjusted_mass = torch.ones(self.vocab_size)
        adjusted_mass[child_symbols] = child_mass
        assert adjusted_mass.shape[0] == node.q_conditionals.shape[0],\
            "Adjusted mass shape {} | Q conditional shape {}"\
            .format(adjusted_mass.shape,
                    node.q_conditionals.shape)

        node.q_conditionals *= adjusted_mass
        node.total_mass = node.q_conditionals.sum()
        node.q_conditionals = node.q_conditionals / node.total_mass

    def _adjust_marginal_probabilities_by_depth(self,depth):

        nodes_to_adjust = self.depth_dict[depth]
        for node in nodes_to_adjust:
            self._adjust_marginal_probabilities_by_node(node)

    def _remove_terminal_depth(self):
        terminal_depth =len(self.depth_sizes)-1
        leaf_nodes = self.depth_dict[terminal_depth]
        del self.depth_dict[terminal_depth]
        for ln in leaf_nodes:
            parent = ln.parent
            del parent.children[ln.symbol]

    def prune(self):
        """
        Prune back tree to only offer options in the
        correct partitioned support
        (and also restructure probabilities)
        """
        leaf_parent_depth = len(self.depth_sizes)-2
        leaf_parents = self.depth_dict[leaf_parent_depth]
        for lp in leaf_parents:
            self._respect_bs_support(lp)
        for i in reversed(range(leaf_parent_depth)):
            self._adjust_marginal_probabilities_by_depth(i)
        self._remove_terminal_depth()

    def sample_sequence(self, seq_len):
        cur_node = self.root
        depth = 0
        log_p_total, log_q_total = 0.0, 0.0
        sample = []
        while depth < seq_len:
            next_step = torch.distributions.Categorical(probs=cur_node.q_conditionals).sample()
            sample.append(next_step.item())
            log_p_total += cur_node.p_conditionals.log()[next_step]
            log_q_total += cur_node.q_conditionals.log()[next_step]
            depth += 1
            if next_step.item() in cur_node.children:
                cur_node = cur_node.children[next_step.item()]
            else:
                break
        
        return log_p_total, log_q_total, cur_node.hidden_state, depth, next_step, sample