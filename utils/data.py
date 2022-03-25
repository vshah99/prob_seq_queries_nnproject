#################################################################################
#
#             Project Title:  Data Loader Utils
#             Author:         Alex Boyd
#             Date:           2022-03-25
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import torch
import random

#################################################################################
#   Function-Class Declaration
#################################################################################

def load_text(file_name):
  with open(file_name, "r") as f:
    text = f.read().strip()

  vocab = set(text)
  vocab.add("<BOS>")  # beginning of sequence
  char_to_id = {v:i for i,v in enumerate(sorted(list(vocab)))}
  id_to_char = {i:v for v,i in char_to_id.items()}

  return {
      "text": text,
      "vocab": vocab,
      "vocab_size": len(vocab),
      "char_to_id": char_to_id,
      "id_to_char": id_to_char,
  }

def process_data(text_dict, batch_size, seq_len, dev=torch.device("cpu"), splits=(0.9, 0.05, 0.05), **dl_args):
  assert(sum(splits) == 1.0)

  ids = [text_dict["char_to_id"][c] for c in text_dict["text"]]
  ids = torch.LongTensor(ids[:-(len(ids) % seq_len)])  # ensure length is even multiple of `seq_len`
  ids = ids.view(-1, seq_len)
  bos_pad = torch.full((ids.shape[0], 1), text_dict["char_to_id"]["<BOS>"])
  ids = torch.cat((bos_pad, ids), dim=-1).to(dev)

  num_seqs = ids.shape[0]
  split_positions = list(range(num_seqs))
  random.shuffle(split_positions)
  train_ids = ids[:int(num_seqs*splits[0]), :]
  valid_ids = ids[int(num_seqs*splits[0]):int(num_seqs*(splits[0]+splits[1])), :]
  test_ids = ids[int(num_seqs*(splits[0]+splits[1])):, :]

  train_dl = torch.utils.data.DataLoader(
      train_ids,
      batch_size=batch_size,
      shuffle=True,
      **dl_args,
  )
  valid_dl = torch.utils.data.DataLoader(
      valid_ids,
      batch_size=batch_size,
      shuffle=False,
      **dl_args,
  )
  test_dl = torch.utils.data.DataLoader(
      test_ids,
      batch_size=batch_size,
      shuffle=False,
      **dl_args,
  )

  return train_dl, valid_dl, test_dl

#################################################################################
#   Main Method
#################################################################################



