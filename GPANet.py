import torch.nn as nn
import torch.nn.functional as F
import torch
from SubLayerConnection import SublayerConnection
from Multihead_Attention import MultiHeadedAttention

class FPM(nn.Module):
    def __init__(self, opts, pad_idx=0):
        # default : num_embeddings = 50265, filter_sizes = [2, 3, 5, 7, 11], hidden_dim = 200, num_channel = 100,
        # num_layers = 6, dropout = 0.3, top_p = 3, h = 2
        super(FPM, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=opts.num_embeddings, embedding_dim=opts.hidden_dim, padding_idx=pad_idx
        )
        self.num_channel = opts.num_channel
        self.filter_sizes = [int(i) for i in opts.filter_sizes.split(',')]
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=self.num_channel,
                                              kernel_size=(filter_size, opts.hidden_dim))
                                    for filter_size in self.filter_sizes])
        self.top_p = opts.top_p

    def forward(self, x, return_indecs=False):
        # [ batch_size, seq_len ]
        x = self.embedding(x).unsqueeze(1)

        # [ batch_size, 1, seq_len, embed_dim ]
        conved = [F.relu(conv(x).squeeze(3)) for conv in self.convs]
        top_p_max = [torch.topk(conv, self.top_p, 2, True, False).values for conv in conved]
        pooled_aver = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        x_max = torch.cat(top_p_max, -1)
        x_avg = torch.cat(pooled_aver, -1)
        x_max = x_max.view(-1, len(self.filter_sizes) * self.top_p, self.num_channel)
        x_avg = x_avg.view(-1, len(self.filter_sizes), self.num_channel)
        if return_indecs:
            return torch.cat([x_max, x_avg], dim=1), [torch.topk(conv, self.top_p, 2, True, False).indices for conv in
                                                      conved]
        else:
            return torch.cat([x_max, x_avg], dim=1) , None

class GPANet(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.num_layers = opts.num_layers
        self.num_channel = opts.num_channel
        self.filter_sizes = len(opts.filter_sizes.split(','))
        self.attention = nn.ModuleList()
        self.sublayer_connection = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.fpm = FPM(opts)
        self.d_model = opts.num_channel
        for _layer in range(self.num_layers):
            self.attention.append(MultiHeadedAttention(h=opts.att_head, d_model=self.d_model))
            self.sublayer_connection.append(SublayerConnection(size=self.d_model, dropout=opts.dropout))
            self.linear_layers.append(nn.Linear(in_features=self.d_model, out_features=self.d_model))
        self.output = nn.Linear(self.num_channel * (self.top_p + 1) * self.filter_sizes, 2)
        self.RNN = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True)
        self.LSTM = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True)
        self.GRU = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=2, batch_first=True)

    def forward(self, x, return_indecs=False, seq_model='Att'):
        assert seq_model in ['Att', 'RNN', 'LSTM', 'GRU']
        x, indecs = self.fpm(x, return_indecs=return_indecs)
        x_res = x
        if seq_model == 'Att':
            for layer_idx in range(self.num_layers):
                x = self.sublayer_connection[layer_idx](x, lambda _x: self.attention[
                    layer_idx].forward(_x, _x, _x))
                x = self.linear_layers[layer_idx](x)
        if seq_model == 'RNN':
            x, _ = self.RNN(x)
        if seq_model == 'LSTM':
            x, _ = self.LSTM(x)
        if seq_model == 'GRU':
            x, _ = self.GRU(x)
        x = (x_res + x) / 2
        if return_indecs:
            return F.softmax(self.output(x.reshape(-1, self.num_channel * (self.top_p + 1) * self.filter_sizes))), indecs
        else:
            return F.softmax(self.output(x.reshape(-1, self.num_channel * (self.top_p + 1) * self.filter_sizes)))
