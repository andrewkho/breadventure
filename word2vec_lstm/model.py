import numpy as np

import torch.nn as nn


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self,
                 rnn_type,
                 init_embed_weights: np.ndarray,
                 nhid,
                 nlayers,
                 dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()

        ntoken, ninp = init_embed_weights.shape
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.encoder.load_state_dict({'weight': init_embed_weights})

        if rnn_type == 'BiLSTM':
            self.rnn = nn.LSTM(ninp,
                               nhid,
                               nlayers,
                               dropout=dropout,
                               bidirectional=True)
            self.decoder = nn.Linear(2*nhid, ntoken)
        elif rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp,
                                             nhid,
                                             nlayers,
                                             dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh',
                                'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(
                    "An invalid option for `--model` was supplied, "
                    "options are ['BiLSTM', 'LSTM', 'GRU', 'RNN_TANH' or "
                    "'RNN_RELU']")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              dropout=dropout)
            self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models"
        # (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for
        # Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if rnn_type == 'BiLSTM':
                print("Warning, BiLSTM can not have tied weights!")
            else:
                if nhid != ninp:
                    raise ValueError('When using the tied flag, nhid must '
                                     'be equal to emsize')
                self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1),
                                           output.size(2)))
        return decoded.view(output.size(0),
                            output.size(1),
                            decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'BiLSTM':
            return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
                    weight.new_zeros(self.nlayers*2, bsz, self.nhid))
        elif self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def _get_param_string(self):
        return f'em{self.nimp}' \
               f'_hid{self.nhid}_la{self.nlayers}_dr{self.drop:.2f}'

    def get_filename(self, prefix):
        return f'{prefix}_{self._get_param_string()}.pt'

    def get_state_filename(self, prefix):
        return f'{prefix}_{self._get_param_string()}_state.pt'
