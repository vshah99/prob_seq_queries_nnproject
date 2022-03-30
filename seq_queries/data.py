random.seed(0)

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

def process_data(text_dict, args): # batch_size, seq_len, dev=torch.device("cpu"), splits=(0.9, 0.05, 0.05), **dl_args):
    tr_split, v_split = args.train_data_pct, args.valid_data_pct  # data split percentages for training and validation
    seq_len, dev = args.seq_len, args.device 

    ids = [text_dict["char_to_id"][c] for c in text_dict["text"]]
    ids = torch.LongTensor(ids[:-(len(ids) % seq_len)])  # ensure length is even multiple of `seq_len`
    ids = ids.view(-1, seq_len)
    bos_pad = torch.full((ids.shape[0], 1), text_dict["char_to_id"]["<BOS>"])
    ids = torch.cat((bos_pad, ids), dim=-1).to(dev)

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