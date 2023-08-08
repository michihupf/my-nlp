import torch
import numpy as np

import torch.cuda as cuda 
import torch.backends.mps as mps
from torch.utils.data import DataLoader

import dataset
from train import compute_loss, LSTM
from const import BATCH_SIZE, SEQ_LEN

device = (
    'cuda' if cuda.is_available() else 
    'mps' if mps.is_available() else 
    'cpu'
)
print(f'{device=}')

(train_data, test_data), (idx_to_tok, tok_to_idx) = dataset.generate_or_load()
print(f'Received data')

def encode(text):
    return [tok_to_idx[t] for t in text]

def decode(embedding):
    return "".join([idx_to_tok[i] for i in embedding])

def score(model, data):
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    
    accs = []
    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            out = model(X)

            # compute indices of output
            idx = torch.max(out, 1)[1]
            accs.append(sum([1 for i, x in enumerate(idx) if x == y[i]]) / len(idx))

    return sum(accs) / len(accs)

def generate_text(model, prompt, num_chars=100):
    prompt = encode(prompt)
    output = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_chars):
            # reshape to (batch_size, seq_len, input_size)
            x = np.reshape(prompt, (1, -1, 1))
            x = torch.tensor(x, dtype=torch.float32).to(device) / len(idx_to_tok)

            out = model(x)

            idx = torch.max(out, 1)[1].cpu().numpy()[0]

            # append to prompt
            prompt.append(idx)
            output.append(idx)

            prompt = prompt[1:]

    return decode(output)


def prompt(text):
    print(text, generate_text(model, text, 100))


model = torch.load('model.pt', map_location=device)
print(f'Loaded model: {model}')
# print(f'Test loss is: {compute_loss(model, test_data)}')

# acc_train = score(model, train_data)
# acc = score(model, test_data)
# print(f'Test accuracy: {acc}, Train accuracy: {acc_train}')

prompt("I am")
