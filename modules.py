import torch
import pytorch_lightning as pl

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class DynamicLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(
            input_size, self.hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, attention_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:-1], device=self.device)

        seq_lens = attention_mask.sum(-1)
        batch_size = attention_mask.shape[0]
        seq_len = attention_mask.shape[1]

        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort.cpu(), batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)
        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        batch_indices = torch.arange(0, batch_size, device=self.device)
        seq_indices = seq_lens - 1
        y_split = y.view(batch_size, length.max(), 2, self.hidden_size)

        output = torch.cat(
            [y_split[batch_indices, seq_indices, 0], y_split[batch_indices, 0, 1]], dim=-1)

        return output


class DynamicGRU(pl.LightningModule):
    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicGRU, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.GRU(
            input_size, self.hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, attention_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones(x.shape[:-1], device=self.device)

        seq_lens = attention_mask.sum(-1)
        batch_size = attention_mask.shape[0]
        seq_len = attention_mask.shape[1]

        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort.cpu(), batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)
        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        batch_indices = torch.arange(0, batch_size, device=self.device)
        seq_indices = seq_lens - 1
        y_split = y.view(batch_size, length.max(), 2, self.hidden_size)

        output = torch.cat(
            [y_split[batch_indices, seq_indices, 0], y_split[batch_indices, 0, 1]], dim=-1)

        return output


import torch.nn as nn

class SimpleTransformer(pl.LightningModule):
    def __init__(self, d_model=100, nhead=8, num_layers=1, dropout=0., pooling='mean'):
        self.pooling = pooling
        super(SimpleTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,  # We don't need the decoder for this task
            dropout=dropout
        )

    def forward(self, x, attention_mask=None):
        # The transformer expects the sequence dimension to be first, so we permute
        x = x.permute(1, 0, 2)
        
        if attention_mask is not None:
            # Transformer's attention mask expects -inf for unwanted positions
            attention_mask = (1.0 - attention_mask) * -1e9  # Assuming attention_mask is 0 for unwanted positions and 1 otherwise

        # Pass through transformer
        output = self.transformer.encoder(x, src_key_padding_mask=attention_mask)
        
        # Restore the original dimension order
        output = output.permute(1, 0, 2)
        
        if self.pooling == 'cls':
            pooled_output = output[:, 0, :]
        elif self.pooling == 'max':
            pooled_output = output.max(dim=1)
        else:
            pooled_output = output.mean(dim=1)

        return pooled_output


# import torch.nn.functional as F
# import math

# class SimpleTransformer(pl.LightningModule):
#     def __init__(self, d_model=100, nhead=8, num_layers=1, dropout=0., pooling='mean'):
#         super(SimpleTransformer, self).__init__()
#         self.d_model = d_model
#         self.pooling = pooling
#         self.transformer = nn.Transformer(
#             d_model=d_model,
#             nhead=nhead,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=0,  # We don't need the decoder for this task
#             dropout=dropout
#         )

#     def forward(self, x, attention_mask=None):
#         # The transformer expects the sequence dimension to be first, so we permute
#         x = x.permute(1, 0, 2)

#         if attention_mask is not None:
#             # Transformer's attention mask expects -inf for unwanted positions
#             attention_mask = (1.0 - attention_mask) * -1e9 

#         # Add positional embeddings to the input embeddings
#         x += self.positional_encoding(x.size(0), self.d_model).to(x.device)

#          # Assuming attention_mask is 0 for unwanted positions and 1 otherwise

#         # Pass through transformer
#         output = self.transformer.encoder(x, src_key_padding_mask=attention_mask)

#         # Restore the original dimension order
#         output = output.permute(1, 0, 2)
        
#         if self.pooling == 'cls':
#             pooled_output = output[:, 0, :]
#         elif self.pooling == 'max':
#             pooled_output = output.max(dim=1).values
#         else:
#             # Assuming attention_mask is 0 for unwanted positions and 1 otherwise
#             pooled_output = self.masked_mean(output, attention_mask)

#         return pooled_output

#     @staticmethod
#     def positional_encoding(seq_len, d_model):
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(seq_len, d_model)
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(1)

#     def masked_mean(self, tensor, mask):
#         """
#         Compute mean of tensor along dimension 1, but ignoring positions with value 0 in the mask.
#         """
#         sum_tensor = (tensor.clone() * mask.unsqueeze(-1)).sum(dim=1)
#         mean_tensor = sum_tensor / mask.sum(dim=1, keepdim=True).clamp_min(1)  # clamp_min to avoid division by 0
#         return mean_tensor