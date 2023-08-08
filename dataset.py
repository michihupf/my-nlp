import numpy as np
import os
import torch
from torch.utils.data import random_split, TensorDataset
from const import INPUT_SIZE, OUTPUT_SIZE, SEQ_LEN
from torch.backends import mps
import sys


def generate_or_load(train_split=0.8, force_regen=False, device='cpu'):
    train_data = None
    test_data = None
    # token mappings
    idx_to_tok = None
    tok_to_idx = None

    if os.path.exists('data/train.pt') \
    and os.path.exists('data/test.pt') \
    and os.path.exists('data/idx.pt') \
    and os.path.exists('data/tok.pt') \
    and not force_regen:
        print(f'Loading data from files')
        train_data = torch.load('data/train.pt', map_location=device)
        test_data = torch.load('data/test.pt', map_location=device)
        idx_to_tok = torch.load('data/idx.pt')
        tok_to_idx = torch.load('data/tok.pt')
    else:
        print(f'Generating dataset')

        ## 16 features + 1 label
        DATA = None

        with open('data/moby_dick.txt', 'r', encoding='utf-8-sig') as f:
            text = f.read()
            idx_to_tok = dict(enumerate(dict.fromkeys(text)))
            tok_to_idx = {token: idx for idx, token in idx_to_tok.items()}

            text = np.array([tok_to_idx[tok] for tok in text])
            print(f'Tokenization done. {len(idx_to_tok)} tokens.')
            if len(idx_to_tok) != OUTPUT_SIZE:
                print(f'WARNING: OUTPUT_SIZE ({OUTPUT_SIZE}) not matching token amount ({len(idx_to_tok)})')

            DATA = np.lib.stride_tricks.sliding_window_view(text, SEQ_LEN+1)

        X, y = DATA[:,0:SEQ_LEN], DATA[:,-1]
        n_patterns = X.shape[0]
        X = torch.tensor(X).float().reshape(n_patterns, SEQ_LEN, INPUT_SIZE).to(device)
        y = torch.tensor(y).long().to(device)

        X /= len(idx_to_tok)

        dataset = TensorDataset(X, y)
        train_data, test_data = random_split(dataset, [train_split, 1-train_split])
        print(f'Split data into training ({len(train_data)} samples) and test ({len(test_data)} samples)')
        torch.save(train_data, 'data/train.pt')
        torch.save(test_data, 'data/test.pt')
        torch.save(idx_to_tok, 'data/idx.pt')
        torch.save(tok_to_idx, 'data/tok.pt')
        print(f'Saved dataset')

    return (train_data, test_data), (idx_to_tok, tok_to_idx)


if __name__ == '__main__':
    force_regen = False
    train_split = 0.8
    if '--rebuild' in sys.argv:
        force_regen = True
    if '--train-split' in sys.argv:
        i = sys.argv.index('--train-split')

    device = (
        'cuda' if torch.cuda.is_available() else 
        'mps' if mps.is_available() else 
        'cpu'
    ) 

    generate_or_load(force_regen=force_regen, device=device)
