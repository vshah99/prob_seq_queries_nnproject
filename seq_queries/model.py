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

#################################################################################
#   Function-Class Declaration
#################################################################################


class CharLM(nn.Module):
    """Character-level Language Model. Handles embedding of sequences of inputs
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

    def forward(self, x, rnn_args=None):
        """Takes in LongTensor `x` of size [batch_size, seq_len] and produces logits
        for next token prediction of size [batch_size, seq_len, vocab_size]."""

        x = F.one_hot(x, num_classes=self.vocab_size).float()
        x, misc_out = self.rnn(x, rnn_args)
        x = self.out_transform(x)

        return {
            "logits": x,
            "misc_output": misc_out
        }

    def graded_forward(self, x):
        """Forward pass during training. Computes self-supervised mean cross-entropy
        loss."""
        x_in = x[..., :-1]
        x_tgt = x[..., 1:]

        output = self.forward(x_in)
        output["loss"] = self.loss_func(
            output["logits"].reshape(-1, self.vocab_size),
            x_tgt.reshape(-1),
        )

        return output

    def get_next_probs(self, x, rnn_args=None, temperature=1.0):
        """Computes the probability distribution over the vocabulary for the next
        term in a sequence. Returns this and resulting hidden state. Can specify a
        temperature to divide the logits by prior to performing a softmax to change
        how 'peaked' or 'flat' the distribution is."""

        step_output = self.forward(x=x, rnn_args=rnn_args)
        logits = step_output["logits"][:, -1, :]  # last position in the sequence
        probs = torch.softmax(logits / temperature, dim=-1)
        return probs, step_output["misc_output"]  # probability tensor size: (batch, vocab), hidden state tensor

    @torch.no_grad()
    def sample(
        self,
        x=None,
        batch_size=None,
        num_steps=1,
        char_to_id=None,
        id_to_char=None,
        temperature=1.0,
    ):
        self.eval()

        # Prep initial input if none is provided to predicate on
        if x is None:
            assert(batch_size is not None)
            assert(char_to_id is not None)
            x = torch.full((batch_size, 1), char_to_id["<BOS>"])
            x = x.to(next(self.parameters()).device)
        batch_size = x.shape[0]

        samples = [x]
        step_input = x
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
        raise NotImplementedError
    else:
        return CharLM(
            vocab_size=args.vocab_size,
            embed_dim=args.hidden_size,
            rnn=rnn,
        )


#################################################################################
#   Main Method
#################################################################################
