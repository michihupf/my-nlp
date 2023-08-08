import numpy as np
from sklearn.model_selection import train_test_split

## token mappings
IDX_TOKENS = None
TOKEN_IDX = None

INPUT_SIZE = 16
## 16 features + 1 label
DATASET = None

with open('data/moby_dick.txt', 'r', encoding='utf-8-sig') as f:
    text = f.read()
    IDX_TOKENS = dict(enumerate(dict.fromkeys(text)))
    TOKEN_IDX = {token: idx for idx, token in IDX_TOKENS.items()}

    text = np.array([TOKEN_IDX[tok] for tok in text])

    DATASET = np.lib.stride_tricks.sliding_window_view(text, INPUT_SIZE+1)

# print(IDX_TOKENS)
# print(TOKEN_IDX)

import torch
import torch.nn as nn 
import torch.optim as optim
import torch.cuda as cuda 
import torch.backends.mps as mps

CUDA = cuda.is_available()
MPS = mps.is_available()
device = (
    'cuda' if CUDA else 
    'mps' if MPS else 
    'cpu'
)

def train(model, X, y, epochs=100, lr=0.1):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        for x, label in zip(X, y):
            out, (hn, cn) = model(x)
            optimizer.zero_grad()
            print(f'{out=}, {hn=}, {cn=}')
            print(f'{label=}')
            print(f'{hn.shape=}, {label.shape=}')
            loss = loss_fn(hn, label)
            loss.backward()
            optimizer.step()



X, y = DATASET[:,0:16], DATASET[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

X_train = torch.tensor(np.expand_dims(X_train, axis=0)).float().to(device)
X_test = torch.tensor(np.expand_dims(X_test, axis=0)).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

print(X_train[0], y_train[0])
print(X_test[0], y_train[0])
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

lr = 0.1
model = nn.Sequential(
          nn.LSTM(input_size=INPUT_SIZE, hidden_size=1, num_layers=1, batch_first=True)
        )
model.to(device)

train(model, X_train, y_train)

