import math

import torch
from torch import nn, Tensor


class LSTMCell(nn.Module):

    """
    An implementation of Hochreiter & Schmidhuber:
    'Long-Short Term Memory' cell.
    http://www.bioinf.jku.at/publications/older/2604.pdf

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.c2c = Tensor(hidden_size * 3)
        self.reset_parameters()



    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        # pdb.set_trace()
        hx, cx = hidden

        x = x.view(-1, x.size(1))

        gates = self.x2h(x) + self.h2h(hx)

        gates = gates.squeeze()

        c2c = self.c2c.unsqueeze(0)
        ci, cf, co = c2c.chunk(3 ,1)
        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate + ci * cx)
        forget_gate = torch.sigmoid(forget_gate + cf * cx)
        cell_gate = forget_gate * cx + in_gate * torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate + co * cell_gate)


        hm = out_gate * F.tanh(cell_gate)
        return (hm, cell_gate)

