import torch
import sys
import random
import pickle as pkl
import pandas as pd
import numpy as np
import pandas as pd

random.seed(0)
from .utils import read_pkl, write_pkl

#######################################################################
# Utilities for loading app data
#######################################################################

def load_amazon_data(data_path):
    text_dict = read_pkl(data_path)
    text_dict['vocab_size'] = len(text_dict['vocab'])
    return text_dict


def process_amazon_data(text_dict, args): # batch_size, seq_len, dev=torch.device("cpu"), splits=(0.9, 0.05, 0.05), **dl_args):
    tr_split, v_split = args.train_data_pct, args.val_data_pct  # data split percentages for training and validation
    seq_len, dev = args.seq_len, args.device

    ids = text_dict['text'].long()

    num_seqs = ids.shape[0]
    split_pos = list(range(num_seqs))
    random.shuffle(split_pos)
    # split into training, validation, and test split tensors
    # train_ids = ids[split_pos[:int(num_seqs*tr_split)], :]
    valid_ids = ids[split_pos[int(num_seqs*tr_split):int(num_seqs*(tr_split+v_split))], :]
    test_ids = ids[split_pos[int(num_seqs*(tr_split+v_split)):], :]

    if args.min_phase_shift:
        phase_shifts = read_pkl(args.dataset_phase_shift_path)
        assert phase_shifts.shape[0] == ids.shape[0],\
            f"Phase shifts was shape {phase_shifts.shape[0]} but ids was shape {ids.shape[0]}"
        # train_ids = train_ids[
        #     phase_shifts[
        #         split_pos[:int(num_seqs*tr_split)]
        #     ] >= args.min_phase_shift, :
        # ]
        valid_ids = valid_ids[
            phase_shifts[
                split_pos[int(num_seqs*tr_split):int(num_seqs*(tr_split+v_split))]
            ] >= args.min_phase_shift,:
        ]
        write_pkl(torch.LongTensor(split_pos[int(num_seqs*tr_split):int(num_seqs*(tr_split+v_split))]),"transition_inds.pkl")

        test_ids = test_ids[
            phase_shifts[
                split_pos[int(num_seqs*(tr_split+v_split)):]
            ] >= args.min_phase_shift, :
        ]
        sys.exit(1)

    train_dl = torch.utils.data.DataLoader(
        train_ids,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_dl, valid_dl, test_dl

#######################################################################
# Main
#######################################################################




def read_mobile_app_data(data_path):
    df = pd.read_csv(data_path, sep='\t')
    df.loc[:,'timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user_id',
                        'session_id',
                        'timestamp'])
    vocab = set(df['app_name'].drop_duplicates().values)
    return df.loc[:,['user_id','app_name']].values, vocab

def stratify_data_by_user(df):
    seqs = []; curr_seq = []
    curr_user, token = df[0]
    curr_seq.append(token)
    for i in range(df.shape[0]):
        user,token = df[i]
        # New user and new session
        if user != curr_user:
            curr_user = user
            seqs.append(curr_seq)
            curr_seq = []
        # Just a new session, same user
        curr_seq.append(token)

    return seqs

def get_user_sequences(df_list,
                      seq_len,
):
    flat_seqs = []
    for user_data in df_list:
        if len(user_data) < seq_len:
            continue
        for i in range(seq_len,len(user_data),1):
            flat_seqs.append(user_data[i-seq_len:i])
    return flat_seqs

def prepare_mobile_app_data_by_user(
    data_path,
    seq_len,
):

    df, vocab = read_mobile_app_data(data_path)
    df_list = stratify_data_by_user(df)
    df_sequences = get_user_sequences(df_list,seq_len)
    return df_sequences, vocab


#######################################################################
# General load information
#######################################################################


def load_text_data(file_name):
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

def load_app_data(file_name, seq_len=15):

    text, vocab = prepare_mobile_app_data_by_user(file_name, seq_len)
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


def process_app_data(text_dict, args): # batch_size, seq_len, dev=torch.device("cpu"), splits=(0.9, 0.05, 0.05), **dl_args):
    tr_split, v_split = args.train_data_pct, args.val_data_pct  # data split percentages for training and validation
    seq_len, dev = args.seq_len, args.device

    ids = [[text_dict["char_to_id"][c] for c in user_data] for user_data in text_dict['text']]
    ids = torch.LongTensor(ids)
    # ids = torch.LongTensor(ids[:-(len(ids) % seq_len)])  # ensure length is even multiple of `seq_len`
    ids = ids.view(-1, seq_len)
    bos_pad = torch.full((ids.shape[0], 1), text_dict["char_to_id"]["<BOS>"])
    ids = torch.cat((bos_pad, ids), dim=-1)

    num_seqs = ids.shape[0]
    split_pos = list(range(num_seqs))
    random.shuffle(split_pos)
    # split into training, validation, and test split tensors
    train_ids = ids[split_pos[:int(num_seqs*tr_split)], :]
    valid_ids = ids[split_pos[int(num_seqs*tr_split):int(num_seqs*(tr_split+v_split))], :]
    test_ids = ids[split_pos[int(num_seqs*(tr_split+v_split)):], :]

    train_dl = torch.utils.data.DataLoader(
        train_ids,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_dl, valid_dl, test_dl

#######################################################################
# Main
#######################################################################



def process_text_data(text_dict, args): # batch_size, seq_len, dev=torch.device("cpu"), splits=(0.9, 0.05, 0.05), **dl_args):
    tr_split, v_split = args.train_data_pct, args.val_data_pct  # data split percentages for training and validation
    seq_len, dev = args.seq_len, args.device

    ids = [text_dict["char_to_id"][c] for c in text_dict["text"]]
    ids = torch.LongTensor(ids[:-(len(ids) % seq_len)])  # ensure length is even multiple of `seq_len`
    ids = ids.view(-1, seq_len)
    bos_pad = torch.full((ids.shape[0], 1), text_dict["char_to_id"]["<BOS>"])
    ids = torch.cat((bos_pad, ids), dim=-1)

    num_seqs = ids.shape[0]
    split_pos = list(range(num_seqs))
    random.shuffle(split_pos)
    # split into training, validation, and test split tensors
    train_ids = ids[split_pos[:int(num_seqs*tr_split)], :]
    valid_ids = ids[split_pos[int(num_seqs*tr_split):int(num_seqs*(tr_split+v_split))], :]
    test_ids = ids[split_pos[int(num_seqs*(tr_split+v_split)):], :]

    train_dl = torch.utils.data.DataLoader(
        train_ids,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_dl = torch.utils.data.DataLoader(
        test_ids,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return train_dl, valid_dl, test_dl

#######################################################################
# Main
#######################################################################

