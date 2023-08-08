import numpy as np
import copy
import torch
import torch.nn as nn 
import torch.optim as optim
import torch.cuda as cuda 
import torch.backends.mps as mps
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR

## token mappings
IDX_TOKENS = None
TOKEN_IDX = None

from const import EPOCHS, INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, OUTPUT_SIZE

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        out, _ = self.lstm(X)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

CUDA = cuda.is_available()
MPS = mps.is_available()
device = (
    'cuda' if CUDA else 
    'mps' if MPS else 
    'cpu'
)
print(f'{device=}')

def train(model, data, epochs=100, lr=0.01, log_freq=128):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    best_model_states = None 
    best_loss = np.inf
    last_loss = 0
    loss_history = []
    model.train()
    print('Starting training of model')

    n_batches = len(data_loader)
    bn_size = len(str(n_batches))
    for e in range(epochs):
        train_loss = 0
        print(50*'-')
        print(f'Epoch: {e+1}\t')
        print(f'Best Loss: {best_loss}; Last Loss: {last_loss}')
        print(50*'-')
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            
            loss = loss_fn(out, y)
            train_loss += loss.item() / n_batches
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % log_freq == 0:
                print(f'Epoch: {e+1}; Batch: {str(i).zfill(bn_size)} / {n_batches}; Loss: {loss}')

    
        last_loss = train_loss
        loss_history.append(train_loss)
        if train_loss < best_loss:
            best_loss = train_loss 
            best_model_states = copy.deepcopy(model.state_dict())

        scheduler.step()

    model.load_state_dict(best_model_states)

    return model, loss_history, best_loss


def compute_loss(model, data):
    data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    test_loss = 0
    model.eval()
    n_batches = len(data_loader)
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            yh = model(X)
            loss = loss_fn(yh, y)

            test_loss += loss.item()

    return test_loss / n_batches

if __name__ == '__main__':
    model = LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, n_layers=3)
    model.to(device)

    import dataset
    (train_data, test_data), (idx_to_tok, tok_to_idx) = dataset.generate_or_load(0.8)
    print(f'Received training data ({len(train_data)} samples) and test data ({len(test_data)} samples).')

    model, loss_history, model_loss = train(model, train_data, epochs=EPOCHS, log_freq=500)
    print(f'Finished training. Loss: {model_loss}')

    test_loss = compute_loss(model, test_data)
    print(f'Test loss is: {test_loss}')

    print(f'Saving...')
    torch.save(model, 'model.pt')
