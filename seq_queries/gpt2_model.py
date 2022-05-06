#################################################################################
#
#             Project Title:  GPT-2 Modeling
#             Author:         Sam Showalter
#             Date:           2022-05-02
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import copy

import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from datasets import load_dataset

from utils import read_pkl, write_pkl

#################################################################################
#   Function-Class Declaration
#################################################################################


class Gpt2ClassificationCollator(object):
    r"""
    Data Collator used for GPT2 in a classificaiton rask.

    It uses a given tokenizer and label encoder to convert any text and labels to numbers that
    can go straight into a GPT2 model.

    This class is built with reusability in mind: it can be used as is as long
    as the `dataloader` outputs a batch in dictionary format that can be passed
    straight into the model - `model(**batch)`.

    Arguments:

      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.

      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to
          labels names and Values map to number associated to those labels.

      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.

    """

    def __init__(self, use_tokenizer, max_sequence_len=None):

        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len

    def __call__(self, sequences):
        r"""
        This function allowes the class objesct to be used as a function call.
        Sine the PyTorch DataLoader needs a collator function, I can use this
        class as a function.

        Arguments:

          item (:obj:`list`):
              List of texts and labels.

        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
          It holddes the statement `model(**Returned Dictionary)`.
        """

        # Get all texts from sequences list.
        texts = [sequence['text'] for sequence in sequences]
        # Get all labels from sequences list.
        # labels = [sequence['label'] for sequence in sequences]
        # # Encode all labels using label encoder.
        # labels = [self.labels_encoder[label] for label in labels]
        # Call tokenizer on all texts to convert into tensors of numbers with
        # appropriate padding.
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        # inputs.update({'labels':torch.tensor(labels)})

        return inputs

def prep_gpt2(batch_size = 128,
              device=0,
              **kwargs):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # default to left padding
    tokenizer.padding_side = "right"
    # Define PAD Token = EOS Token = 50256
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_special_tokens({
    # "bos_token": "<s>",
    # "unk_token": "<unk>",
    # "pad_token": "<pad>",
    # "mask_token": "<mask>"
    # })
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
        # res = model.forward(input_ids = data['input_ids'], attention_mask=data['attention_mask'])
    # dataset = load_dataset("wikitext",'wikitext-2-v1', split="validation")
    dataset = load_dataset("bookcorpus", split="validation[:1%]")
    sys.exit(1)
    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer)
    # print(dataset.column_names)
    # dataset = dataset.map(lambda e: tokenizer(e['text']))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, collate_fn=gpt2_classificaiton_collator)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        model.to(device)
        for data in dataloader:
            print(data)
            # print(torch.Tensor(data['input_ids']))
            res = model(**data)
            # res = model.forward(input_ids = data['input_ids'], attention_mask=data['attention_mask'])
        return {
            "model":model,
            "tokenizer": tokenizer,
            "dataset":dataset,
        }

res = prep_gpt2()
print("HI")


#################################################################################
#   Main Method
#################################################################################



