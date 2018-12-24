"""
Example RNN model code stolen from
https://github.com/pytorch/examples/blob/master/word_language_model/model.py
"""
import numpy as np
from torch import nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self,
                 rnn_type,
                 ntoken,
                 ninp,
                 nhid,
                 nlayers,
                 dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearities = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}
                nonlinearity = nonlinearities[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model`
                                 was supplied, options are
                                 ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(
                ninp,
                nhid,
                nlayers,
                nonlinearity=nonlinearity,
                dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        if tie_weights:
            if nhid != ninp:
                raise ValueError("""When using the tied flag,
                                 nhid must be equal to emsize""")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def predict(self, input, hidden):
        """ Forward pass through encoder, rnn, decoder"""
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), \
                hidden

    def forward(self, input, targets, hidden):
        outputs, hidden = self.predict(input, hidden)
        outputs = outputs.view(-1, outputs.size()[-1])
        return F.cross_entropy(outputs, targets), hidden

    def diagnose(self, input, targets, hidden):
        """ Perform forward pass and calculate loss + perplexity"""
        outputs, hidden = self.predict(input, hidden)
        outputs = outputs.view(-1, outputs.size()[-1])
        loss = F.cross_entropy(outputs, targets).item()
        perplexity = np.exp(loss)
        return {'loss': loss, 'perplexity': perplexity}, hidden

    def init_hidden(self, bsz):
        """ Initialize the hidden state for each layer """
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
