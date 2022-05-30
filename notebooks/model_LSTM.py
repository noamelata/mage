import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


def normalize(x):
    eps = 1E-6
    return x / (x.norm(dim=1, keepdim=True) + eps)


def split_by_4(x):
    return torch.split(x, x.shape[0] // 4)


def combine_batch(new, old):
    if new.shape == old.shape:
        return new
    b = new.shape[0]
    return torch.cat([new, torch.zeros_like(old[b:])], dim=0) + torch.cat([torch.zeros_like(old[:b]), old[b:]], dim=0)


def apply_fwd_grad(dFg, vw):
    return torch.matmul(dFg.permute(1, 0), vw.view(vw.shape[0], -1)).view(vw.shape[1], vw.shape[2])

def create_new_vs(x_part, h_part, shapes, i, f, g, o, c_t_1, tanh_c_t, device, epsilon):
    pw_ii = torch.randn((shapes['W_ii'], 1), device=device) * epsilon
    pw_if = torch.randn((shapes['W_if'], 1), device=device) * epsilon
    pw_ig = torch.randn((shapes['W_ig'], 1), device=device) * epsilon
    pw_io = torch.randn((shapes['W_io'], 1), device=device) * epsilon
    pw_hi = torch.randn((shapes['W_hi'], 1), device=device) * epsilon
    pw_hf = torch.randn((shapes['W_hf'], 1), device=device) * epsilon
    pw_hg = torch.randn((shapes['W_hg'], 1), device=device) * epsilon
    pw_ho = torch.randn((shapes['W_ho'], 1), device=device) * epsilon
    _vb_ii = torch.randn(shapes['b_ii'], device=device) * epsilon
    _vb_if = torch.randn(shapes['b_if'], device=device) * epsilon
    _vb_ig = torch.randn(shapes['b_ig'], device=device) * epsilon
    _vb_io = torch.randn(shapes['b_io'], device=device) * epsilon
    _vb_hi = torch.randn(shapes['b_hi'], device=device) * epsilon
    _vb_hf = torch.randn(shapes['b_hf'], device=device) * epsilon
    _vb_hg = torch.randn(shapes['b_hg'], device=device) * epsilon
    _vb_ho = torch.randn(shapes['b_ho'], device=device) * epsilon

    _vw_ii = torch.matmul(pw_ii, normalize(x_part).unsqueeze(1))
    _vw_ii = (o * (1 - (tanh_c_t ** 2)) * g * (i * (1 - i))).unsqueeze(2).expand(_vw_ii.shape) * _vw_ii
    _vw_hi = torch.matmul(pw_hi, normalize(h_part).unsqueeze(1))
    _vw_hi = (o * (1 - (tanh_c_t ** 2)) * g * (i * (1 - i))).unsqueeze(2).expand(_vw_hi.shape) * _vw_hi
    _vw_if = torch.matmul(pw_if, normalize(x_part).unsqueeze(1))
    _vw_if = (o * (1 - (tanh_c_t ** 2)) * c_t_1 * (f * (1 - f))).unsqueeze(2).expand(
        _vw_if.shape) * _vw_if
    _vw_hf = torch.matmul(pw_hf, normalize(h_part).unsqueeze(1))
    _vw_hf = (o * (1 - (tanh_c_t ** 2)) * c_t_1 * (f * (1 - f))).unsqueeze(2).expand(
        _vw_hf.shape) * _vw_hf
    _vw_ig = torch.matmul(pw_ig, normalize(x_part).unsqueeze(1))
    _vw_ig = (o * (1 - (tanh_c_t ** 2)) * i * (1 - (g ** 2))).unsqueeze(2).expand(_vw_ig.shape) * _vw_ig
    _vw_hg = torch.matmul(pw_hg, normalize(h_part).unsqueeze(1))
    _vw_hg = (o * (1 - (tanh_c_t ** 2)) * i * (1 - (g ** 2))).unsqueeze(2).expand(_vw_hg.shape) * _vw_hg
    _vw_io = torch.matmul(pw_io, normalize(x_part).unsqueeze(1))
    _vw_io = (tanh_c_t * (o * (1 - o))).unsqueeze(2).expand(_vw_io.shape) * _vw_io
    _vw_ho = torch.matmul(pw_ho, normalize(h_part).unsqueeze(1))
    _vw_ho = (tanh_c_t * (o * (1 - o))).unsqueeze(2).expand(_vw_ho.shape) * _vw_ho

    _vb_ii = (o * (1 - (tanh_c_t ** 2)) * g * (i * (1 - i))) * _vb_ii
    _vb_hi = (o * (1 - (tanh_c_t ** 2)) * g * (i * (1 - i))) * _vb_hi
    _vb_if = (o * (1 - (tanh_c_t ** 2)) * c_t_1 * (f * (1 - f))) * _vb_if
    _vb_hf = (o * (1 - (tanh_c_t ** 2)) * c_t_1 * (f * (1 - f))) * _vb_hf
    _vb_ig = (o * (1 - (tanh_c_t ** 2)) * i * (1 - (g ** 2))) * _vb_ig
    _vb_hg = (o * (1 - (tanh_c_t ** 2)) * i * (1 - (g ** 2))) * _vb_hg
    _vb_io = (tanh_c_t * (o * (1 - o))) * _vb_io
    _vb_ho = (tanh_c_t * (o * (1 - o))) * _vb_ho

    return(_vw_ii, _vw_hi, _vw_if, _vw_hf, _vw_ig, _vw_hg, _vw_io, _vw_ho,
           _vb_ii, _vb_hi, _vb_if, _vb_hf, _vb_ig, _vb_hg, _vb_io, _vb_ho)


class RNN(nn.Module):
    def __init__(self, rnn_type, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx, train_embedding=True):
        super().__init__()
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Cell states
        if rnn_type == 'RNN':
            self.keys_rnn = ["c"]
        elif rnn_type == 'GRU':
            self.keys_rnn = ["r", "z", "c"]  # c = n, but we need this label.
        elif rnn_type == 'LSTM':
            self.keys_rnn = ["i", "f", "c", "o"]  # c = g.
        else:
            raise NotImplementedError()
        # Number of states
        self.n_states = len(self.keys_rnn)

        # Number of directions
        n_directions = 1 + int(bidirectional)
        self.n_directions = n_directions
        # Total number of distinguishable weights
        self.n_rnn = self.n_states * n_directions

        # Define layers
        # Embedding. Set padding_idx so that the <pad> is not updated. 
        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # RNN
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers,
                              bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                               bidirectional=bidirectional, dropout=dropout)
        # Decoder: fully-connected
        self.decoder = nn.Linear(hidden_dim * n_directions, output_dim)

        # Train the embedding?
        if not train_embedding:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Setup dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, batch_text):
        text, text_lengths = batch_text
        # text = [sentence len, batch size]
        embedded = self.encoder(text)
        # embedded = [sent len, batch size, emb dim]

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        if self.rnn_type == 'LSTM':
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
        else:
            packed_output, hidden = self.rnn(packed_embedded)
        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]
        # Unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [sentence len, batch size, hid dim * num directions]

        # Concatenate the final hidden layers (for bidirectional)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
            (-1, self.hidden_dim * self.n_directions))

        # Dropout
        hidden = self.drop(hidden)

        # Decode
        decoded = self.decoder(hidden).squeeze(1)

        return decoded

    def fwd_mode(self, batch_text, y, loss):
        text, text_lengths = batch_text
        # text = [sentence len, batch size]
        embedded = self.encoder(text)
        # embedded = [sent len, batch size, emb dim]

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        if isinstance(packed_embedded, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = packed_embedded
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        num_directions = 2 if self.rnn.bidirectional else 1
        zeros = torch.zeros(self.rnn.num_layers * num_directions,
                            max_batch_size, self.rnn.hidden_size,
                            dtype=input.dtype, device=input.device)
        hx = (zeros, zeros)
        self.rnn.check_forward_args(input, hx, batch_sizes)

        x = torch.split(input, tuple(batch_sizes))
        device = input.device
        epsilon = 1
        V = {}
        grad = 0
        h_stack = []
        c_stack = []
        with torch.no_grad():
            for j in range(self.rnn.num_layers):
                h, c_t_1 = hx[j]
                h_list = [h]
                h_grad_list = []
                h_full_batch = torch.zeros_like(h)
                accumulated_grad = None

                W_ii, W_if, W_ig, W_io = split_by_4(self.rnn.__getattr__(f"weight_ih_l{j}"))
                W_hi, W_hf, W_hg, W_ho = split_by_4(self.rnn.__getattr__(f"weight_hh_l{j}"))
                b_ii, b_if, b_ig, b_io = split_by_4(self.rnn.__getattr__(f"bias_ih_l{j}"))
                b_hi, b_hf, b_hg, b_ho = split_by_4(self.rnn.__getattr__(f"bias_hh_l{j}"))

                vw_ii = 0
                vw_hi = 0
                vw_if = 0
                vw_hf = 0
                vw_ig = 0
                vw_hg = 0
                vw_io = 0
                vw_ho = 0
                vb_ii = 0
                vb_hi = 0
                vb_if = 0
                vb_hf = 0
                vb_ig = 0
                vb_hg = 0
                vb_io = 0
                vb_ho = 0
                dh_t_dW = None
                dc_dW = None
                dc_dh_t_1 = None

                for seq in range(len(x)):
                    x_t = x[seq]
                    # if x_t.shape[0] != 64:
                    #     continue
                    h_part = h[:x_t.shape[0]]
                    c_t_1 = c_t_1[:x_t.shape[0]]
                    dz_dW = z_grad_list[seq] if j > 0 else None

                    _vw_ii = torch.randn(W_ii.shape, device=device) * epsilon
                    _vw_if = torch.randn(W_if.shape, device=device) * epsilon
                    _vw_ig = torch.randn(W_ig.shape, device=device) * epsilon
                    _vw_io = torch.randn(W_io.shape, device=device) * epsilon
                    _vw_hi = torch.randn(W_hi.shape, device=device) * epsilon
                    _vw_hf = torch.randn(W_hf.shape, device=device) * epsilon
                    _vw_hg = torch.randn(W_hg.shape, device=device) * epsilon
                    _vw_ho = torch.randn(W_ho.shape, device=device) * epsilon
                    _vb_ii = torch.randn(b_ii.shape, device=device) * epsilon
                    _vb_if = torch.randn(b_if.shape, device=device) * epsilon
                    _vb_ig = torch.randn(b_ig.shape, device=device) * epsilon
                    _vb_io = torch.randn(b_io.shape, device=device) * epsilon
                    _vb_hi = torch.randn(b_hi.shape, device=device) * epsilon
                    _vb_hf = torch.randn(b_hf.shape, device=device) * epsilon
                    _vb_hg = torch.randn(b_hg.shape, device=device) * epsilon
                    _vb_ho = torch.randn(b_ho.shape, device=device) * epsilon


                    i_p = x_t @ W_ii.T + b_ii + h_part @ W_hi.T + b_hi
                    di_p_dW = x_t @ _vw_ii.T + (dz_dW @ W_ii.T if dz_dW is not None else 0) + _vb_ii + h_part @ _vw_hi.T + (dh_t_dW[:x_t.shape[0]] @ W_hi.T if dh_t_dW is not None else 0) + _vb_hi
                    di_p_dh_t_1 = W_hi
                    i = torch.sigmoid(i_p)
                    di_dW = (i * (1 - i)) * di_p_dW
                    di_dh_t_1 = (i * (1 - i)).unsqueeze(-1) * di_p_dh_t_1

                    f_p = x_t @ W_if.T + b_if + h_part @ W_hf.T + b_hf
                    df_p_dW = x_t @ _vw_if.T + (dz_dW @ W_if.T if dz_dW is not None else 0) + _vb_if + h_part @ _vw_hf.T + (dh_t_dW[:x_t.shape[0]] @ W_hf.T if dh_t_dW is not None else 0) + _vb_hf
                    df_p_dh_t_1 = W_hf
                    f = torch.sigmoid(f_p)
                    df_dW = (f * (1 - f)) * df_p_dW
                    df_dh_t_1 = (f * (1 - f)).unsqueeze(-1) * df_p_dh_t_1


                    g_p = x_t @ W_ig.T + b_ig + h_part @ W_hg.T + b_hg
                    dg_p_dW = x_t @ _vw_ig.T + (dz_dW @ W_ig.T if dz_dW is not None else 0) + _vb_ig + h_part @ _vw_hg.T + (dh_t_dW[:x_t.shape[0]] @ W_hg.T if dh_t_dW is not None else 0) + _vb_hg
                    dg_p_dh_t_1 = W_hg
                    g = torch.tanh(g_p)
                    dg_dW = (1 - g ** 2) * dg_p_dW
                    dg_dh_t_1 = (1 - g ** 2).unsqueeze(-1) * dg_p_dh_t_1

                    o_p = x_t @ W_io.T + b_io + h_part @ W_ho.T + b_ho
                    do_p_dW = x_t @ _vw_io.T + (dz_dW @ W_io.T if dz_dW is not None else 0) + _vb_io + h_part @ _vw_ho.T + (dh_t_dW[:x_t.shape[0]] @ W_ho.T if dh_t_dW is not None else 0) + _vb_ho
                    do_p_dh_t_1 = W_ho
                    o = torch.sigmoid(o_p)
                    do_dW = (o * (1 - o)) * do_p_dW
                    do_dh_t_1 = (o * (1 - o)).unsqueeze(-1) * do_p_dh_t_1

                    c_t = f * c_t_1 + i * g
                    dc_dW = df_dW * c_t_1 + (dc_dW[:x_t.shape[0]] * f if dc_dW is not None else 0) + di_dW * g + dg_dW * i
                    dc_dh_t_1 = df_dh_t_1 * c_t_1.unsqueeze(-1) + (dc_dh_t_1[:x_t.shape[0]] * f.unsqueeze(-1) if dc_dh_t_1 is not None else 0) + di_dh_t_1 * g.unsqueeze(-1) + dg_dh_t_1 * i.unsqueeze(-1)

                    tanh_c_t = torch.tanh(c_t)
                    d_tanh_c_t_dW = dc_dW * (1 - tanh_c_t ** 2)
                    d_tanh_c_t_dh_t_1 = dc_dh_t_1 * (1 - tanh_c_t ** 2).unsqueeze(-1)

                    h = o * tanh_c_t
                    dh_t_dW = do_dW * tanh_c_t + d_tanh_c_t_dW * o
                    dh_t_dh_t_1 = do_dh_t_1 * tanh_c_t.unsqueeze(-1) + d_tanh_c_t_dh_t_1 * o.unsqueeze(-1)

                    if accumulated_grad is None:
                        accumulated_grad = dh_t_dW
                    else:
                        accumulated_grad = combine_batch((dh_t_dh_t_1 @ accumulated_grad[:x_t.shape[0]].unsqueeze(
                                -1)).squeeze() + dh_t_dW, accumulated_grad)

                    h_list.append(h)
                    h_full_batch = combine_batch(h, h_full_batch)
                    h_grad_list.append(dh_t_dW)
                    c_t_1 = c_t

                    vw_ii += _vw_ii
                    vw_hi += _vw_hi
                    vw_if += _vw_if
                    vw_hf += _vw_hf
                    vw_ig += _vw_ig
                    vw_hg += _vw_hg
                    vw_io += _vw_io
                    vw_ho += _vw_ho
                    vb_ii += _vb_ii
                    vb_hi += _vb_hi
                    vb_if += _vb_if
                    vb_hf += _vb_hf
                    vb_ig += _vb_ig
                    vb_hg += _vb_hg
                    vb_io += _vb_io
                    vb_ho += _vb_ho


                # todo: add dropout as in nn.LSTM
                x = tuple(h_list[1:])
                z_grad_list = h_grad_list
                vw_ih = torch.cat([vw_ii, vw_if, vw_ig, vw_io], dim=0)
                vw_hh = torch.cat([vw_hi, vw_hf, vw_hg, vw_ho], dim=0)
                vb_ih = torch.cat([vb_ii, vb_if, vb_ig, vb_io], dim=0)
                vb_hh = torch.cat([vb_hi, vb_hf, vb_hg, vb_ho], dim=0)
                V[j] = (vw_ih, vw_hh, vb_ih, vb_hh)
                h_stack.append(h_full_batch)
                c_stack.append(c_t)

                grad = grad + accumulated_grad

        packed_output, (hidden, _) = x, (torch.stack(h_stack, dim=0), torch.zeros_like(c_stack[0]))

        self.rnn.permute_hidden(hidden, unsorted_indices)

        hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
            (-1, self.hidden_dim * self.n_directions))

        # # Dropout
        # hidden = self.drop(hidden)

        vw = torch.randn(self.decoder.weight.shape, device=device, dtype=torch.float32) * epsilon
        vb = torch.randn(self.decoder.bias.shape, device=device, dtype=torch.float32) * epsilon
        new_grad = (hidden @ vw.T).squeeze() + vb
        grad = torch.matmul(grad, self.decoder.weight.permute(1, 0)).squeeze() + new_grad

        with torch.no_grad():
            # Decode
            decoded = self.decoder(hidden).squeeze(1)

        dLdout = torch.zeros_like(decoded)

        out = torch.autograd.Variable(decoded, requires_grad=True)
        out.grad = torch.zeros_like(out)
        L = loss(out, y)

        L.backward()
        ##import pdb; pdb.set_trace()
        dLdout = out.grad

        ##grad_transfer = dLdout.permute(1, 0) ## Batch x n_classes
        ##tot_norm = torch.sqrt(tot_norm)

        dFg = (dLdout * grad).sum()

        for i in range(self.rnn.num_layers):
            for w in [self.rnn.__getattr__(f"weight_ih_l{i}"),
                      self.rnn.__getattr__(f"weight_hh_l{i}"),
                      self.rnn.__getattr__(f"bias_ih_l{i}"),
                      self.rnn.__getattr__(f"bias_hh_l{i}")]:
                if w.grad is None:
                    w.grad = torch.zeros_like(w)

            vw_ih, vw_hh, vb_ih, vb_hh = V[i]
            self.rnn.__getattr__(f"weight_ih_l{i}").grad += dFg * vw_ih
            self.rnn.__getattr__(f"weight_hh_l{i}").grad += dFg * vw_hh
            self.rnn.__getattr__(f"bias_ih_l{i}").grad += dFg * vb_ih
            self.rnn.__getattr__(f"bias_hh_l{i}").grad += dFg * vb_hh
        if self.decoder.weight.grad is None:
            self.decoder.weight.grad = torch.zeros_like(self.decoder.weight)
        if self.decoder.bias.grad is None:
            self.decoder.bias.grad = torch.zeros_like(self.decoder.bias)
        self.decoder.weight.grad += dFg * vw
        self.decoder.bias.grad += dFg * vb

        return decoded

    # def fwd_mode(self, batch_text, y, loss):
    #     text, text_lengths = batch_text
    #     # text = [sentence len, batch size]
    #     embedded = self.encoder(text)
    #     # embedded = [sent len, batch size, emb dim]
    #
    #     # Pack sequence
    #     packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
    #
    #     if isinstance(packed_embedded, PackedSequence):
    #         input, batch_sizes, sorted_indices, unsorted_indices = packed_embedded
    #         max_batch_size = batch_sizes[0]
    #         max_batch_size = int(max_batch_size)
    #     num_directions = 2 if self.rnn.bidirectional else 1
    #     zeros = torch.zeros(self.rnn.num_layers * num_directions,
    #                         max_batch_size, self.rnn.hidden_size,
    #                         dtype=input.dtype, device=input.device)
    #     hx = (zeros, zeros)
    #     self.rnn.check_forward_args(input, hx, batch_sizes)
    #
    #     x = torch.split(input, tuple(batch_sizes))
    #     device = input.device
    #     epsilon = 1
    #     V = {}
    #     grad = 0
    #     h_stack = []
    #     c_stack = []
    #     with torch.no_grad():
    #         for j in range(self.rnn.num_layers):
    #             h, c_t_1 = hx[j]
    #             h_list = [h]
    #             accumulated_grad = grad
    #
    #             W_ii, W_if, W_ig, W_io = split_by_4(self.rnn.__getattr__(f"weight_ih_l{j}"))
    #             W_hi, W_hf, W_hg, W_ho = split_by_4(self.rnn.__getattr__(f"weight_hh_l{j}"))
    #             b_ii, b_if, b_ig, b_io = split_by_4(self.rnn.__getattr__(f"bias_ih_l{j}"))
    #             b_hi, b_hf, b_hg, b_ho = split_by_4(self.rnn.__getattr__(f"bias_hh_l{j}"))
    #
    #             vw_ii, vw_hi, vw_if, vw_hf, vw_ig, vw_hg, vw_io, vw_ho = 0, 0, 0, 0, 0, 0, 0, 0
    #             vb_ii, vb_hi, vb_if, vb_hf, vb_ig, vb_hg, vb_io, vb_ho = 0, 0, 0, 0, 0, 0, 0, 0
    #
    #             for seq in range(len(x)):
    #                 x_part = x[seq]
    #                 h_part = h[:x_part.shape[0]]
    #                 c_t_1 = c_t_1[:x_part.shape[0]]
    #
    #                 i = torch.sigmoid(x_part @ W_ii.T + b_ii + h_part @ W_hi.T + b_hi)
    #                 f = torch.sigmoid(x_part @ W_if.T + b_if + h_part @ W_hf.T + b_hf)
    #                 g = torch.tanh(x_part @ W_ig.T + b_ig + h_part @ W_hg.T + b_hg)
    #                 o = torch.sigmoid(x_part @ W_io.T + b_io + h_part @ W_ho.T + b_ho)
    #                 c_t = f * c_t_1 + i * g
    #                 tanh_c_t = torch.tanh(c_t)
    #
    #                 shapes = {'W_ii': W_ii.shape[0],
    #                           'W_if': W_if.shape[0],
    #                           'W_ig': W_ig.shape[0],
    #                           'W_io': W_io.shape[0],
    #                           'W_hi': W_hi.shape[0],
    #                           'W_hf': W_hf.shape[0],
    #                           'W_hg': W_hg.shape[0],
    #                           'W_ho': W_ho.shape[0],
    #                           'b_ii': b_ii.shape,
    #                           'b_if': b_if.shape,
    #                           'b_ig': b_ig.shape,
    #                           'b_io': b_io.shape,
    #                           'b_hi': b_hi.shape,
    #                           'b_hf': b_hf.shape,
    #                           'b_hg': b_hg.shape,
    #                           'b_ho': b_ho.shape,
    #                           }
    #
    #                 _vw_ii, _vw_hi, _vw_if, _vw_hf, _vw_ig, _vw_hg, _vw_io, _vw_ho, _vb_ii, _vb_hi, _vb_if, _vb_hf, _vb_ig, _vb_hg, _vb_io, _vb_ho = create_new_vs(x_part, h_part, shapes, i, f, g, o, c_t_1, tanh_c_t, device, epsilon)
    #
    #                 new_grad_x = torch.matmul(_vw_ii, x_part.unsqueeze(2)).squeeze() + _vb_ii + \
    #                              torch.matmul(_vw_if, x_part.unsqueeze(2)).squeeze() + _vb_if + \
    #                              torch.matmul(_vw_ig, x_part.unsqueeze(2)).squeeze() + _vb_ig + \
    #                              torch.matmul(_vw_io, x_part.unsqueeze(2)).squeeze() + _vb_io
    #                 new_grad_h = torch.matmul(_vw_hi, h_part.unsqueeze(2)).squeeze() + _vb_hi + \
    #                              torch.matmul(_vw_hf, h_part.unsqueeze(2)).squeeze() + _vb_hf + \
    #                              torch.matmul(_vw_hg, h_part.unsqueeze(2)).squeeze() + _vb_hg + \
    #                              torch.matmul(_vw_ho, h_part.unsqueeze(2)).squeeze() + _vb_ho
    #
    #                 if h_part.shape == h.shape:
    #
    #                     accumulated_grad = accumulated_grad * new_grad_h + new_grad_x
    #
    #                     h = o * tanh_c_t
    #                     c_t_1 = c_t
    #                     h_list.append(h)
    #
    #                     vw_ii += _vw_ii
    #                     vw_hi += _vw_hi
    #                     vw_if += _vw_if
    #                     vw_hf += _vw_hf
    #                     vw_ig += _vw_ig
    #                     vw_hg += _vw_hg
    #                     vw_io += _vw_io
    #                     vw_ho += _vw_ho
    #                     vb_ii += _vb_ii
    #                     vb_hi += _vb_hi
    #                     vb_if += _vb_if
    #                     vb_hf += _vb_hf
    #                     vb_ig += _vb_ig
    #                     vb_hg += _vb_hg
    #                     vb_io += _vb_io
    #                     vb_ho += _vb_ho
    #
    #                 else:
    #                     accumulated_grad = combine_batch(accumulated_grad[:h_part.shape[0]] * new_grad_h + new_grad_x,
    #                                                      accumulated_grad)
    #
    #                     h = combine_batch(o * tanh_c_t, h)
    #                     c_t_1 = c_t
    #                     h_list.append(h)
    #
    #                     vw_ii += combine_batch(_vw_ii, torch.zeros_like(vw_ii))
    #                     vw_hi += combine_batch(_vw_hi, torch.zeros_like(vw_hi))
    #                     vw_if += combine_batch(_vw_if, torch.zeros_like(vw_if))
    #                     vw_hf += combine_batch(_vw_hf, torch.zeros_like(vw_hf))
    #                     vw_ig += combine_batch(_vw_ig, torch.zeros_like(vw_ig))
    #                     vw_hg += combine_batch(_vw_hg, torch.zeros_like(vw_hg))
    #                     vw_io += combine_batch(_vw_io, torch.zeros_like(vw_io))
    #                     vw_ho += combine_batch(_vw_ho, torch.zeros_like(vw_ho))
    #                     vb_ii += combine_batch(_vb_ii, torch.zeros_like(vb_ii))
    #                     vb_hi += combine_batch(_vb_hi, torch.zeros_like(vb_hi))
    #                     vb_if += combine_batch(_vb_if, torch.zeros_like(vb_if))
    #                     vb_hf += combine_batch(_vb_hf, torch.zeros_like(vb_hf))
    #                     vb_ig += combine_batch(_vb_ig, torch.zeros_like(vb_ig))
    #                     vb_hg += combine_batch(_vb_hg, torch.zeros_like(vb_hg))
    #                     vb_io += combine_batch(_vb_io, torch.zeros_like(vb_io))
    #                     vb_ho += combine_batch(_vb_ho, torch.zeros_like(vb_ho))
    #
    #             # todo: add dropout as in nn.LSTM
    #             x = torch.stack(h_list[1:])
    #             vw_ih = torch.cat([vw_ii, vw_if, vw_ig, vw_io], dim=1)
    #             vw_hh = torch.cat([vw_hi, vw_hf, vw_hg, vw_ho], dim=1)
    #             vb_ih = torch.cat([vb_ii, vb_if, vb_ig, vb_io], dim=1)
    #             vb_hh = torch.cat([vb_hi, vb_hf, vb_hg, vb_ho], dim=1)
    #             V[j] = (vw_ih, vw_hh, vb_ih, vb_hh)
    #             h_stack.append(h)
    #             c_stack.append(c_t)
    #
    #             grad += accumulated_grad
    #
    #     packed_output, (hidden, _) = x, (torch.stack(h_stack, dim=0), torch.zeros_like(c_stack[0]))
    #
    #     self.rnn.permute_hidden(hidden, unsorted_indices)
    #
    #     hidden = (torch.transpose(hidden[-self.n_directions:], 0, 1)).reshape(
    #         (-1, self.hidden_dim * self.n_directions))
    #
    #     # # Dropout
    #     # hidden = self.drop(hidden)
    #
    #     vn = torch.randn([self.decoder.weight.shape[0], 1], device=device, dtype=torch.float32) * epsilon
    #     vb = vn.clone().squeeze().expand(self.decoder.bias.shape)
    #     vw = torch.matmul(vn, normalize(hidden).unsqueeze(1))
    #     new_grad = torch.matmul(vw, hidden.unsqueeze(2)).squeeze() + vb
    #     grad = torch.matmul(grad, self.decoder.weight.permute(1, 0)) + new_grad
    #
    #     with torch.no_grad():
    #         # Decode
    #         decoded = self.decoder(hidden).squeeze(1)
    #
    #     dLdout = torch.zeros_like(decoded)
    #
    #     out = torch.autograd.Variable(decoded, requires_grad=True)
    #     out.grad = torch.zeros_like(out)
    #     L = loss(out, y)
    #
    #     L.backward()
    #     ##import pdb; pdb.set_trace()
    #     dLdout = out.grad
    #
    #     ##grad_transfer = dLdout.permute(1, 0) ## Batch x n_classes
    #     ##tot_norm = torch.sqrt(tot_norm)
    #
    #     dFg = (dLdout * grad).sum(1, keepdim=True)
    #
    #     for i in range(self.rnn.num_layers):
    #         for w in [self.rnn.__getattr__(f"weight_ih_l{i}"),
    #                   self.rnn.__getattr__(f"weight_hh_l{i}"),
    #                   self.rnn.__getattr__(f"bias_ih_l{i}"),
    #                   self.rnn.__getattr__(f"bias_hh_l{i}")]:
    #             if w.grad is None:
    #                 w.grad = torch.zeros_like(w)
    #
    #         vw_ih, vw_hh, vb_ih, vb_hh = V[i]
    #         self.rnn.__getattr__(f"weight_ih_l{i}").grad += apply_fwd_grad(dFg, vw_ih)
    #         self.rnn.__getattr__(f"weight_hh_l{i}").grad += apply_fwd_grad(dFg, vw_hh)
    #         self.rnn.__getattr__(f"bias_ih_l{i}").grad += torch.sum(dFg * vb_ih, dim=0)
    #         self.rnn.__getattr__(f"bias_hh_l{i}").grad += torch.sum(dFg * vb_hh, dim=0)
    #     if self.decoder.weight.grad is None:
    #         self.decoder.weight.grad = torch.zeros_like(self.decoder.weight)
    #     if self.decoder.bias.grad is None:
    #         self.decoder.bias.grad = torch.zeros_like(self.decoder.bias)
    #     self.decoder.weight.grad += apply_fwd_grad(dFg, vw)
    #     self.decoder.bias.grad += dFg.sum() * vb
    #
    #     return decoded
