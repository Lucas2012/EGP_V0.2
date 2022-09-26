import numpy as np
import torch
from torch import nn
from collections import OrderedDict

from models.modules import ContextOnlySoftDotAttention, SoftDotAttention, VisualSoftDotAttention

class CustomRNN(nn.Module):
    """
    A module that runs multiple steps of RNN cell
    With this module, you can use mask for variable-length input
    """
    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(CustomRNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, mask, hx, reset_indices=None, return_c=False):
        max_time = input_.size(0)
        output = []
        output_c = []
        for time in range(max_time):
            h_next, c_next = cell(input_[time], hx=hx)
            mask_ = mask[time].unsqueeze(1).expand_as(h_next)
            h_next = h_next*mask_ + hx[0]*(1 - mask_)
            c_next = c_next*mask_ + hx[1]*(1 - mask_)
            output.append(h_next)
            if return_c:
              output_c.append(c_next)
            # Reset memory for samples that just had a period if applicable. 
            # Done AFTER appending to output so we don't have zero vectors in output.
            if reset_indices is not None:
              reset_ = reset_indices[time].unsqueeze(1).expand_as(h_next)
            else:
              reset_ = 0
            h_next = h_next * (1 - reset_)
            c_next = c_next * (1 - reset_)
            hx_next = (h_next, c_next)
            hx = hx_next
        output = torch.stack(output, 0)
        if len(output_c) > 0:
          output_c = torch.stack(output_c, 0)
        return output, output_c, hx

    def forward(self, input_, mask, hx=None, reset_indices=None, return_c=False):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
            mask = mask.transpose(0, 1)
            if reset_indices is not None:
                reset_indices = reset_indices.transpose(0, 1)
        max_time, batch_size, _ = input_.size()

        if hx is None:
            hx = input_.new(batch_size, self.hidden_size).zero_()
            hx = (hx, hx)
        h_n = []
        c_n = []
        layer_output = None
        layer_output_c = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, layer_output_c, (layer_h_n, layer_c_n) = CustomRNN._forward_rnn(
                cell=cell, input_=input_, mask=mask, hx=hx, reset_indices=reset_indices, return_c=return_c)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        output_c = layer_output_c
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)
        if return_c:
            return output, output_c, (h_n, c_n)
        else:
            return output, (h_n, c_n)


###########################################################################
#                  Neural Machine Translation modules                     #
###########################################################################

class TranslatorEncoderRNN(nn.Module):
    """ Encodes navigation instructions, returning hidden state context (for
        attention methods) and a decoder initial state. """

    def __init__(self, vocab_size, embedding_size, hidden_size, padding_idx,
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

    def forward(self, inputs, lengths, reset_indices, return_c=False):
        """
        Expects input vocab indices as (batch, seq_len). Also requires a list of lengths for dynamic batching.
        """
        embeds = self.embedding(inputs)   # (batch, seq_len, embedding_size)
        embeds = self.drop(embeds)

        embeds_mask = self.create_mask(embeds.size(0), embeds.size(1), lengths)
        if self.bidirectional:
            results_1 = self.rnn(embeds, mask=embeds_mask, reset_indices=reset_indices, return_c=return_c)
            output_1, (ht_1, ct_1) = results_1[0], results_1[-1]
            results_2 = self.rnn(self.flip(embeds, 1), mask=self.flip(embeds_mask, 1), reset_indices=self.flip(reset_indices, 1), return_c=return_c)
            output_2, (ht_2, ct_2) = results_2[0], results_2[-1]
            output = torch.cat((output_1, self.flip(output_2, 0)), 2)
            if return_c:
                output_c = torch.cat((results_1[1], self.flip(results_2[1], 0)), 2)
            ht = torch.cat((ht_1, ht_2), 2)
            ct = torch.cat((ct_1, ct_2), 2)
        else:
            results = self.rnn(embeds, mask=embeds_mask, reset_indices=reset_indices, return_c=return_c)
            output, (ht, ct) = results[0], results[-1]
            if return_c:
                output_c = results[1]
        if return_c:
            return output.transpose(0, 1), output_c.transpose(0, 1), ht.squeeze(), ct.squeeze(), embeds_mask
        else:
            return output.transpose(0, 1), ht.squeeze(), ct.squeeze(), embeds_mask


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, hidden_size,
                 dropout_ratio, use_linear=False, share_embedding=False, use_end_token=False):
        super(DecoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        if use_linear:
            self.embedding = nn.Linear(self.vocab_size, vocab_embedding_size, bias=False)
        else:
            self.embedding = nn.Embedding(self.vocab_size, vocab_embedding_size)
        self.drop = nn.Dropout(p=dropout_ratio)
        self.lstm = nn.LSTMCell(vocab_embedding_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)
        word_embeds = word_embeds.squeeze()  # (batch, embedding_size)
        word_embeds_drop = self.drop(word_embeds)

        h_1, c_1 = self.lstm(word_embeds_drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit


###########################################################################
#                        Speaker model modules                            #
#              Lifted directly from speaker-follower code                 #
###########################################################################
class SpeakerEncoderRNN(nn.Module):
    def __init__(self, max_node_in_path, neuralese_len, img_feat_input_dim, hidden_size, dropout_ratio, bidirectional=False):
        super(SpeakerEncoderRNN, self).__init__()
        assert not bidirectional, 'Bidirectional is not implemented yet'

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_node_in_path           = max_node_in_path
        self.neuralese_len              = neuralese_len
        self.img_feat_input_dim         = img_feat_input_dim
        self.hidden_dim                 = hidden_size
        self.drop                       = nn.Dropout(p=dropout_ratio)
        self.visual_attention_layer     = VisualSoftDotAttention(
                                          hidden_size, img_feat_input_dim)
        self.lstm                       = nn.LSTMCell(
                                          2 * img_feat_input_dim, hidden_size)
        self.encoder2decoder            = nn.Linear(
                                          hidden_size, hidden_size)
        

    def _tensorfy_img_features(self, all_img_feats):
        batch_size = len(all_img_feats)
        length = np.zeros(len(all_img_feats), np.uint8)
        img_feats_tensor = [
            np.zeros((batch_size, 36, self.img_feat_input_dim), np.float32) 
            for _ in range(self.max_node_in_path)]
        for i, img_feats in enumerate(all_img_feats):
            length[i] = len(img_feats)
            for t, img_feat in enumerate(img_feats):
                img_feats_tensor[t][i] = img_feat
        img_feats_tensor = [
            torch.from_numpy(img_feats).to(self.device) 
            for img_feats in img_feats_tensor]
        return img_feats_tensor, self._create_mask(batch_size, length)
    
    def _tensorfy_action_embeddings(self, all_action_embeddings):
        batch_size = len(all_action_embeddings)
        length = np.zeros(len(all_action_embeddings), np.uint8)
        action_embeddings_tensor = [
            np.zeros((batch_size, self.img_feat_input_dim), np.float32) 
            for _ in range(self.max_node_in_path)]
        for i, action_embeddings in enumerate(all_action_embeddings):
            length[i] = len(action_embeddings)
            for t, action_emb in enumerate(action_embeddings):
                action_embeddings_tensor[t][i] = action_emb
        action_embeddings_tensor = [
            torch.from_numpy(action_embeddings).to(self.device) 
            for action_embeddings in action_embeddings_tensor]
        return action_embeddings_tensor, self._create_mask(batch_size, length)

    def _create_mask(self, batch_size, length):
        tensor_mask = torch.zeros(batch_size, self.max_node_in_path)
        for idx, row in enumerate(tensor_mask):
            row[:length[idx]] = 1
        return tensor_mask.to(self.device)
    
    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = torch.zeros(batch_size, self.hidden_dim)
        c0 = torch.zeros(batch_size, self.hidden_dim)
        return h0.to(self.device), c0.to(self.device)

    def _forward_one_step(self, h_0, c_0, img_feats, action_embedding):
        feature, _ = self.visual_attention_layer.forward(h_0, img_feats)
        concat_input = torch.cat((action_embedding, feature), 1)
        drop = self.drop(concat_input)
        h_1, c_1 = self.lstm(drop, (h_0, c_0))
        return h_1, c_1
    
    def forward(self, all_img_feats, all_action_embeddings, return_c=False, segment=[False, False, 0]):
        batch_size = len(all_img_feats)
        img_feats_tensor, ctx_mask = self._tensorfy_img_features(all_img_feats)
        action_embeddings_tensor, _ = self._tensorfy_action_embeddings(all_action_embeddings)
        h, c = self.init_state(batch_size)
        h_list = []
        c_list = []

        # whether to segment and the probability of reset
        if segment[0]:
            reset_indices = []              
        h_final = torch.zeros(h.shape).to(self.device)
        for t, img_feats in enumerate(img_feats_tensor):
            action_emb = action_embeddings_tensor[t]
            h, c = self._forward_one_step(
                h, c, img_feats, action_emb)
            if segment[0]:
                reset_samples = torch.rand(img_feats.shape[0]).to(self.device)
                reset_index   = (reset_samples < segment[-1]).float()
                reset_indices.append(reset_index.unsqueeze(0))
                if segment[1]:
                    reset_ = reset_index.unsqueeze(1).expand_as(h)
                else:
                    reset_ = 0
            else:
                reset_ = 0
            h = h * (1 - reset_)
            c = c * (1 - reset_)
            for i in range(batch_size):
                if ctx_mask[i,t] == 1:
                    h_final[i] = h[i]
            h_list.append(h)
            if return_c:
                c_list.append(c)

        # decoder_init = nn.Tanh()(self.encoder2decoder(h))
        decoder_init = nn.Tanh()(self.encoder2decoder(h_final))

        ctx = torch.stack(h_list, dim=1)  # (batch, seq_len, hidden_size)
        ctx = self.drop(ctx)
        outputs = [ctx]
        if return_c:
            ctx_c = torch.stack(c_list, dim=1)
            outputs.append(ctx_c)

        outputs += [decoder_init, c, ctx_mask]

        assert(c.max()==c.max())

        if segment[0]:
            reset_indices = torch.cat(reset_indices, dim=0).transpose(0,1)
            outputs.append(reset_indices)
        return outputs


# PHASING OUT OF USE. REPLACED WITH DecoderRNN
class SpeakerDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, hidden_size,
                 dropout_ratio):
        super(SpeakerDecoderLSTM, self).__init__()
        self.vocab_size           = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size          = hidden_size
        self.embedding       = nn.Linear(self.vocab_size, vocab_embedding_size, bias=False)
        self.drop            = nn.Dropout(p=dropout_ratio)
        self.lstm            = nn.LSTMCell(vocab_embedding_size, hidden_size)
        self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action  = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)
        word_embeds = word_embeds.squeeze()  # (batch, embedding_size)
        word_embeds_drop = self.drop(word_embeds)

        h_1, c_1 = self.lstm(word_embeds_drop, (h_0, c_0))
        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
        logit = self.decoder2action(h_tilde)
        return h_1, c_1, alpha, logit
