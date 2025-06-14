"""
sLSTM: Scalar Long Short-Term Memory

This module implements the sLSTM (scalar LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The sLSTM extends the traditional LSTM by using exponential gating and a new memory mixing technique,
allowing for improved performance on various sequence modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTM(nn.Module):
    """
        sLSTM layer implementation.

        This layer applies multiple sLSTM cells in sequence, with optional dropout between layers.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden state.
            num_layers (int): Number of sLSTM layers.
            dropout (float, optional): Dropout probability between layers. Default: 0.0.
        """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([
            sLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        seq_length, batch_size, _ = input_seq.size()

        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        # Convert hidden_state to list of tuples
        hidden_state = [
            (hidden_state[0][layer_idx], hidden_state[1][layer_idx], hidden_state[2][layer_idx])
            for layer_idx in range(self.num_layers)
        ]

        outputs = []
        for t in range(seq_length):
            x = input_seq[t, :, :]
            new_hidden_state = []
            for layer_idx, layer in enumerate(self.layers):
                h, c, n = hidden_state[layer_idx]
                h, c, n = layer(x, (h, c, n))
                new_hidden_state.append((h, c, n))
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            hidden_state = new_hidden_state
            outputs.append(x)

        output_seq = torch.stack(outputs, dim=0)

        # Convert hidden_state back to the original format
        hn = torch.stack([h for h, c, n in hidden_state], dim=0)
        cn = torch.stack([c for h, c, n in hidden_state], dim=0)
        nn = torch.stack([n for h, c, n in hidden_state], dim=0)

        return output_seq, (hn, cn, nn)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.layers[0].weight_ih.device)
        )

class sLSTMCell(nn.Module):
    """
        sLSTM cell implementation.

        This cell uses exponential gating as described in the xLSTM paper.

        Args:
            input_size (int): Size of input features.
            hidden_size (int): Size of hidden state.
        """
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        nn.init.zeros_(self.bias)

    def forward(self, input, hx):
        """
                Forward pass of the sLSTM cell.

                Args:
                    input (Tensor): Input tensor of shape (batch_size, input_size).
                    hx (tuple of Tensors): Previous hidden state and cell state.

                Returns:
                    tuple: New hidden state and cell state.
                """
        h, c, n = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        i, f, g, o = gates.chunk(4, 1)
        i = torch.exp(i)  # Exponential input gate
        f = torch.sigmoid(f)  # Sigmoid forget gate
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        n = f * n + i

        c = f * c + i * g
        h = o * (c / (n + 1e-8))  # Avoid division by zero

        return h, c, n

# # 测试与验证
# embedding_dim = 32
# h_dim = 64
# num_layers = 1
# dropout = 0.0
# batch_size = 64
#
# # 创建输入数据
# input_data = torch.randn(8, batch_size, embedding_dim).cuda()  # (seq_length, batch_size, input_size)
#
# # 自定义 sLSTM
# slstm = sLSTM(embedding_dim, h_dim, num_layers, dropout=dropout).cuda()
# state_tuple = slstm.init_hidden(batch_size)
# output_s, hidden_state = slstm(input_data, state_tuple)
# print("Output shape from sLSTM:", output_s.shape)
# print("Hidden state shapes from sLSTM:", [h.shape for h in hidden_state])