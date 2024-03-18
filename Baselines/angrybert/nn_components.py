import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch
import math
import numpy as np

class Word_Embedding(nn.Module): 
    def __init__(self, ntoken, emb_dim=300, dropout=0.0):
        super(Word_Embedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self):
        glove_weight = torch.from_numpy(np.load('embedding_matrix.npy'))
        self.emb.weight.data[:self.ntoken] = glove_weight

    def forward(self, x):
        emb = self.emb(x)  # x should be a tensor
        emb = self.dropout(emb)
        return emb


class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(myLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Select the output from the last time step
        # output = out[:, -1, :].clone()  # Clone the tensor to make it out of place operation
        # print(out[:, -1, :].shape)

        return out[:, -1, :]


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layer = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layer)

    def forward(self, x):
        logits = self.main(x)
        return logits


class Gate_Attention(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(Gate_Attention, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):
        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        output = gate * a + (1 - gate) * b
        return output  # Clone the tensor to make it out of place operation
