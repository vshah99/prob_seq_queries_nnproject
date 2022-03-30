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

    def get_next_probs(self, src, rnn_args=None, temperature=1.0):
        """Computes the probability distribution over the vocabulary for the next
        term in a sequence. Returns this and resulting hidden state. Can specify a
        temperature to divide the logits by prior to performing a softmax to change
        how 'peaked' or 'flat' the distribution is."""

        step_output = self.forward(src=src, rnn_args=rnn_args)
        logits = step_output["logits"][:, -1, :]  # last position in the sequence
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs, step_output["misc_output"]  # probability tensor size: (batch, vocab), hidden state tensor

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
        batch_size = x.shape[0]

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
