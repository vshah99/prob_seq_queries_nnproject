#################################################################################
#
#             Project Title:  Inference Utilities
#             Author:         Alex Boyd
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from collections import OrderedDict
import torch


#################################################################################
#   Function-Class Declaration
#################################################################################

# Compute distribution for next token conditioned on a provided string.
@torch.no_grad()
def next_token_dist(model, str_input, temperature, char_to_id, id_to_char, device = "cuda:0"):
  # Get model device
  dev = next(model.parameters()).device  # Get model device

  # Process input
  id_input = [char_to_id["<BOS>"]] + [char_to_id[c] for c in str_input]
  id_input = torch.Tensor([id_input], device=device, dtype=torch.long)

  # Get probabilities and decode them
  model.eval()
  probs, _ = model.get_next_probs(id_input, rnn_args=None, temperature=temperature)
  probs = probs.squeeze(dim=0)  # squeeze because batch_size of 1

  # sort: most probable to least
  decoded_probs = [(c,probs[i].item()) for i,c in id_to_char.items()]
  decoded_probs = OrderedDict(sorted(decoded_probs, key=lambda x: -x[1]))

  return {
      "most_likely": id_to_char[torch.argmax(probs).item()],
      "prob_tensor": probs,
      "decoded_probs": decoded_probs,
  }

#################################################################################
#   Main Method
#################################################################################



