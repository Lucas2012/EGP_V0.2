import torch
import torch.nn as nn
from collections import OrderedDict

from models.rnn import CustomRNN


class EncoderRNN(nn.Module):
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(self, opts, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1, use_linear=False,
                            share_embedding=False):
        super(EncoderRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts        
        self.embedding_size = embedding_size
        self.share_embedding = share_embedding
        if self.opts.no_language_feats:
            print('Warning: language information is not used for navigation.')
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        if use_linear:
            self.embedding = nn.Linear(vocab_size, embedding_size, bias=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.bidirectional = bidirectional
        self.rnn_kwargs = OrderedDict([
            ('cell_class', nn.LSTMCell),
            ('input_size', embedding_size),
            ('hidden_size', hidden_size),
            ('num_layers', num_layers),
            ('batch_first', True),
            ('dropout', 0),
        ])
        self.rnn = CustomRNN(**self.rnn_kwargs)

    def create_mask(self, batchsize, max_length, length):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(self.device)

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, inputs, lengths):
        """
        Expects input vocab indices as (batch, seq_len). Also requires a list of lengths for dynamic batching.
        """
        share_embedding = self.share_embedding
        if share_embedding is False:
          embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
          embeds = self.drop(embeds)
        else:
          embeds = inputs

        embeds_mask = self.create_mask(embeds.size(0), embeds.size(1), lengths)

        if self.bidirectional:
            output_1, (ht_1, ct_1) = self.rnn(embeds, mask=embeds_mask)
            output_2, (ht_2, ct_2) = self.rnn(self.flip(embeds, 1), mask=self.flip(embeds_mask, 1))
            output = torch.cat((output_1, self.flip(output_2, 0)), 2)
            ht = torch.cat((ht_1, ht_2), 2)
            ct = torch.cat((ct_1, ct_2), 2)
        else:
            output, (ht, ct) = self.rnn(embeds, mask=embeds_mask)

        if self.opts.no_language_feats: # Return values of language encoder are set to zero if we don't use language
            output = torch.zeros_like(output)
            ht     = torch.zeros_like(ht)
            ct     = torch.zeros_like(ct)
        return output.transpose(0, 1), ht.squeeze(), ct.squeeze(), embeds_mask


class TranslatorEncoderRNN(nn.Module):
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(self, opts, vocab_size, embedding_size, hidden_size, padding_idx,
                            dropout_ratio, bidirectional=False, num_layers=1, use_linear=False):
        super(TranslatorEncoderRNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embedding_size = embedding_size
        hidden_size = hidden_size // 2 if bidirectional else hidden_size
        self.num_layers = num_layers
        if use_linear:
            self.embedding = nn.Linear(vocab_size, embedding_size, bias=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.bidirectional = bidirectional
        self.rnn_kwargs = OrderedDict([
            ('cell_class', nn.LSTMCell),
            ('input_size', embedding_size),
            ('hidden_size', hidden_size),
            ('num_layers', num_layers),
            ('batch_first', True),
            ('dropout', 0),
        ])
        self.rnn = CustomRNN(**self.rnn_kwargs)

    def create_mask(self, batchsize, max_length, length):
        """Given the length create a mask given a padded tensor"""
        tensor_mask = torch.zeros(batchsize, max_length)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(self.device)

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def forward(self, inputs, lengths):
        """
        Expects input vocab indices as (batch, seq_len). Also requires a list of lengths for dynamic batching.
        """
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)

        embeds_mask = self.create_mask(embeds.size(0), embeds.size(1), lengths)

        if self.bidirectional:
            output_1, (ht_1, ct_1) = self.rnn(embeds, mask=embeds_mask)
            output_2, (ht_2, ct_2) = self.rnn(self.flip(embeds, 1), mask=self.flip(embeds_mask, 1))
            output = torch.cat((output_1, self.flip(output_2, 0)), 2)
            ht = torch.cat((ht_1, ht_2), 2)
            ct = torch.cat((ct_1, ct_2), 2)
        else:
            output, (ht, ct) = self.rnn(embeds, mask=embeds_mask)

        return output.transpose(0, 1), ht.squeeze(), ct.squeeze(), embeds_mask
