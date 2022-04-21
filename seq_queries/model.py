#################################################################################
#
#             Project Title:  Character-level LMs
#             Author:         Alex Boyd
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from abc import ABC, abstractmethod
from .utils import _tup_cpu

#################################################################################
#   Function-Class Declaration
#################################################################################


class LM(ABC):

    @abstractmethod
    def forward(self, src, **kwargs):
        pass

    @abstractmethod
    def graded_forward(self, src, tgt, **kwargs):
        pass

    @abstractmethod
    def get_next_probs(self, src, rnn_args, temperature, **kwargs):
        pass


class CausalLM(LM, nn.Module):
    """Causal Language Model. Handles embedding of sequences of inputs
    from a fixed vocabulary, processes them through a provided RNN, and then
    produces resulting logits for next token distribution."""

    def __init__(
        self,
        vocab_size,
        embed_dim,
        rnn,
    ):
        super().__init__()

        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.rnn = rnn
        self.out_transform = nn.Linear(embed_dim, vocab_size, bias=True)
        self.loss_func = nn.CrossEntropyLoss()
        self.temperature = None

    def forward(self, src, rnn_args=None, **kwargs):
        """Takes in LongTensor `src` of size [batch_size, seq_len] and produces logits
        for next token prediction of size [batch_size, seq_len, vocab_size]."""

        one_hot_src = F.one_hot(src, num_classes=self.vocab_size).float()
        rnn_out, misc_out = self.rnn(one_hot_src, rnn_args)
        logits = self.out_transform(rnn_out)

        return {
            "logits": logits,
            "misc_output": misc_out
        }

    def graded_forward(self, src, tgt=None, **kwargs):
        """Forward pass during training. Computes self-supervised mean cross-entropy
        loss."""
        if tgt is None:
            x_in = src[..., :-1]
            x_tgt = src[..., 1:]
        else:
            x_in, x_tgt = src, tgt

        output = self.forward(x_in)
        output["loss"] = self.loss_func(
            output["logits"].reshape(-1, self.vocab_size),
            x_tgt.reshape(-1),
        )

        return output

    def get_next_probs(self, x, rnn_args=None, temperature=1.0,
                         max_batch_size=128, device='cpu', return_logits=True, **kwargs):
        """Computes the probability distribution over the vocabulary for the next
        term in a sequence. Returns this and resulting hidden state. Can specify a
        temperature to divide the logits by prior to performing a softmax to change
        how 'peaked' or 'flat' the distribution is."""

        # Model temperature always defaults to 1, must set to None to overwrite
        if self.temperature is not None:
            temperature = self.temperature
        xs = torch.split(x,max_batch_size)
        if rnn_args is not None:
            if isinstance(rnn_args,tuple):
                # Need to split up hidden states
                assert x.shape[0] == rnn_args[0].shape[1] == rnn_args[1].shape[1],\
                    f"Sizes were x: {x.shape}, rnn1 {rnn_args[0].shape}, rnn2 {rnn_args[1].shape}"
                rnn_args = list(zip(*(torch.split(rnn_args[0], max_batch_size, dim =1),
                                torch.split(rnn_args[1],max_batch_size, dim =1))))
            else: rnn_args = torch.split(rnn_args,max_batch_size)
        else: rnn_args = [None]*len(xs)

        prob_outputs = []
        step_outputs = []
        for x, rnn_arg, in zip(xs, rnn_args):
            if isinstance(rnn_arg,tuple):
                rnn_arg_cuda = (rnn_arg[0].to(device),rnn_arg[1].to(device))
                assert x.shape[0] == rnn_arg[0].shape[1] == rnn_arg[1].shape[1],\
                    f"Sizes were x: {x.shape[0]}, rnn1 {rnn_arg[0].shape[1]}, rnn2 {rnn_arg[1].shape[1]}"
            elif rnn_arg is not None: rnn_arg.to(device)
            step_output = self.forward(src=x.to(device), rnn_args=rnn_arg_cuda if rnn_arg else None)
            logits = step_output["logits"][:, -1, :] / temperature # last position in the sequence
            if not return_logits:
                probs = torch.softmax(logits, dim=-1)
                prob_outputs.append(probs.cpu())
            else:
                prob_outputs.append(logits.cpu())
            step_outputs.append(_tup_cpu(step_output['misc_output']))

        # If we have a LSTM
        if isinstance(step_outputs[0], tuple):
            hidden, context = zip(*step_outputs)
            hidden = torch.cat(hidden, dim = 1)
            context = torch.cat(context, dim = 1)
            step_output = (hidden, context)
        # If we just have a single hidden state
        else:
            step_output = torch.cat(step_outputs,dim = 1)

        return torch.cat(prob_outputs,dim = 0), step_output

    @torch.no_grad()
    def sample(
        self,
        src=None,
        batch_size=None,
        num_steps=1,
        char_to_id=None,
        id_to_char=None,
        temperature=1.0,
    ):
        raise NotImplementedError  # TODO: Replace with other sampling functionality. Original code kept here for reference.

        self.eval()

        # Prep initial input if none is provided to predicate on
        if src is None:
            assert(batch_size is not None)
            assert(char_to_id is not None)
            src = torch.full((batch_size, 1), char_to_id["<BOS>"])
            src = src.to(next(self.parameters()).device)

        samples = [src]
        step_input = src
        hidden_state = None
        for _ in range(num_steps):
            probs, hidden_state = self.get_next_probs(step_input, hidden_state, temperature)
            next_token_dist = torch.distributions.categorical.Categorical(probs=probs)
            step_input = next_token_dist.sample().unsqueeze(-1)
            samples.append(step_input)

        samples = torch.cat(samples, dim=-1)
        output = {"sampled_ids": samples}

        # If mapping is provided, convert sampled IDs into actual strings
        if id_to_char is not None:
            strs = []
            samples = samples.cpu()
            for i in range(batch_size):
                strs.append("".join([id_to_char[j.item()] for j in samples[i, :]]))
            output["sampled_strs"] = strs

        return output


class MaskedLM(LM, nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        rnn,
    ):
        super().__init__()

        self.vocab_size, self.embed_dim = vocab_size, embed_dim
        self.rnn = rnn
        self.temperature = None
        self.out_transform = nn.Linear(embed_dim, vocab_size, bias=True)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, src, **kwargs):
        pass

    def graded_forward(self, src, tgt, **kwargs):
        pass

    def get_next_probs(self, src, rnn_args, temperature, **kwargs):
        pass




SUPPORTED_RNN = {
    "RNN": nn.RNN,
    "GRU": nn.GRU,
    "LSTM": nn.LSTM,
}


def get_model(args):
    rnn = SUPPORTED_RNN[args.rnn_type](
        input_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bias=True,
        batch_first=True,
        dropout=args.dropout,
        bidirectional=args.masked_lm,
    )

    if args.masked_lm:
        model = MaskedLM(
            vocab_size=args.vocab_size,
            embed_dim=args.hidden_size,
            rnn=rnn,
        )
    else:
        model = CausalLM(
            vocab_size=args.vocab_size,
            embed_dim=args.hidden_size,
            rnn=rnn,
        )

    model.to(args.device)
    return model


#################################################################################
#   Main Method
#################################################################################
