import torch
import torch.nn as nn
from collections import OrderedDict

from models.modules import build_mlp, SoftAttention, PositionalEncoding, ScaledDotProductAttention, create_mask, proj_masking, PositionalEncoding, VariationalEncoder, EnvDropSoftDotAttention 


class CoGrounding(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16, return_value=False, use_VE=False):
        super(CoGrounding, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.max_len = max_len
        self.return_value = return_value
        self.use_VE = use_VE

        proj_input_dim = img_feat_input_dim
        if self.use_VE:
            self.img_feat_dim = 2048
            self.heading_dim = img_feat_input_dim - self.img_feat_dim 
            self.var_enc = VariationalEncoder(self.img_feat_dim,  opts.ve_hidden_dims, learn_prior = opts.learn_prior)
            proj_input_dim = opts.ve_hidden_dims[-1] + self.heading_dim
        proj_navigable_kwargs = OrderedDict([
            ('input_dim', proj_input_dim),
            ('hidden_dims', img_fc_dim),
            ('use_batchnorm', img_fc_use_batchnorm),
            ('dropout', img_dropout),
            ('fc_bias', fc_bias),
            ('relu', opts.mlp_relu)
        ])
        self.proj_navigable_mlp = build_mlp(**proj_navigable_kwargs)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)
        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=fc_bias)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(img_fc_dim[-1] * 2 + rnn_hidden_size, rnn_hidden_size)

        self.lang_position = PositionalEncoding(rnn_hidden_size, dropout=0.1, max_len=max_len)
        if opts.replace_ctx_w_goal: # If we are replacing ctx, we don't want to add positional encoding.
            self.lang_position = nn.Sequential()

        self.logit_fc = nn.Linear(rnn_hidden_size * 2, img_fc_dim[-1])

        self.h2_fc_lstm = nn.Linear(rnn_hidden_size + img_fc_dim[-1], rnn_hidden_size, bias=fc_bias)

        # value estimation
        if self.return_value:
            self.critic = nn.Sequential(
                nn.Linear(max_len + rnn_hidden_size, 1),
                nn.Tanh()
            )

        self.num_predefined_action = 1

    def _forward_ve(self, feats):
        img_feats, head_feats = torch.split(feats, self.img_feat_dim, dim=-1)
        img_feats, kl_loss = self.var_enc(img_feats)
        feats = torch.cat([img_feats, head_feats], dim=-1)
        return feats, kl_loss

    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
                navigable_index=None, ctx_mask=None, return_action_selector=False):
        """ Takes a single step in the decoder

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size

        pre_feat: previous attended feature, batch x feature_size

        question: this should be a single vector representing instruction

        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        index_length = [len(_index) + self.num_predefined_action for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        if self.use_VE:
            navigable_feat, kl_loss_1 = self._forward_ve(navigable_feat)
            pre_feat, kl_loss_2       = self._forward_ve(pre_feat)
            loss = kl_loss_1 + kl_loss_2
        else:
            loss = 0

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)
        proj_pre_feat = self.proj_navigable_mlp(pre_feat)
        positioned_ctx = self.lang_position(ctx)

        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_0), positioned_ctx, mask=ctx_mask)

        weighted_img_feat, img_attn = self.soft_attn(self.h0_fc(h_0), proj_navigable_feat, mask=navigable_mask)

        # merge info into one LSTM to be carry through time
        concat_input = torch.cat((proj_pre_feat, weighted_img_feat, weighted_ctx), 1)

        h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
        h_1_drop = self.dropout(h_1)

        # policy network
        h_tilde = self.logit_fc(torch.cat((weighted_ctx, h_1_drop), dim=1))
        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        # value estimation
        if self.return_value:
            concat_value_input = self.h2_fc_lstm(torch.cat((h_0, weighted_img_feat), 1))
            h_1_value = self.dropout(torch.sigmoid(concat_value_input) * torch.tanh(c_1))
            value = self.critic(torch.cat((ctx_attn, h_1_value), dim=1))
            results = [h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, value, navigable_mask, loss]
        else:
            results = [h_1, c_1, weighted_ctx, img_attn, ctx_attn, logit, navigable_mask, loss]
        if return_action_selector:
            results += [h_tilde]
        return results

class SpeakerFollowerBaseline(nn.Module):
    """ An unrolled LSTM with attention over instructions for decoding navigation actions. """

    def __init__(self, opts, img_fc_dim, img_fc_use_batchnorm, img_dropout, img_feat_input_dim,
                 rnn_hidden_size, rnn_dropout, max_len, fc_bias=True, max_navigable=16, use_VE=False):
        super(SpeakerFollowerBaseline, self).__init__()

        self.max_navigable = max_navigable
        self.feature_size = img_feat_input_dim
        self.hidden_size = rnn_hidden_size
        self.use_VE = use_VE
        proj_input_dim = img_feat_input_dim
        if self.use_VE:
            self.img_feat_dim = 2048
            self.heading_dim  = img_feat_input_dim - self.img_feat_dim
            self.var_enc = VariationalEncoder(self.img_feat_dim, opts.ve_hidden_dims, learn_prior=opts.learn_prior)
            proj_input_dim = opts.ve_hidden_dims[-1] + self.heading_dim
        self.proj_img_mlp = nn.Linear(proj_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.proj_navigable_mlp = nn.Linear(proj_input_dim, img_fc_dim[-1], bias=fc_bias)

        self.h0_fc = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=False)

        self.soft_attn = SoftAttention()

        self.dropout = nn.Dropout(p=rnn_dropout)

        self.lstm = nn.LSTMCell(proj_input_dim* 2, rnn_hidden_size)

        self.h1_fc = nn.Linear(rnn_hidden_size, rnn_hidden_size, bias=False)

        self.proj_out = nn.Linear(rnn_hidden_size, img_fc_dim[-1], bias=fc_bias)


    def _forward_ve(self, feats):
        img_feats, head_feats = torch.split(feats, self.img_feat_dim, dim=-1)
        img_feats, kl_loss = self.var_enc(img_feats)
        feats = torch.cat([img_feats, head_feats], dim=-1)
        return feats, kl_loss

    #def forward(self, img_feat, navigable_feat, pre_feat, h_0, c_0, ctx, navigable_index=None, ctx_mask=None):
    def forward(self, img_feat, navigable_feat, pre_feat, question, h_0, c_0, ctx, pre_ctx_attend,
                      navigable_index=None, ctx_mask=None):

        """ Takes a single step in the decoder LSTM.

        img_feat: batch x 36 x feature_size
        navigable_feat: batch x max_navigable x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        navigable_index: list of list
        ctx_mask: batch x seq_len - indices to be masked
        """
        batch_size, num_imgs, feat_dim = img_feat.size()

        # add 1 because the navigable index yet count in "stay" location
        # but navigable feature does include the "stay" location at [:,0,:]
        index_length = [len(_index)+1 for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        if self.use_VE:
            img_feat, kl_loss_1       = self._forward_ve(img_feat)
            navigable_feat, kl_loss_2 = self._forward_ve(navigable_feat)
            pre_feat, kl_loss_3       = self._forward_ve(pre_feat)
            loss = kl_loss_1 + kl_loss_2 + kl_loss_3
        else:
            loss = 0

        proj_img_feat = proj_masking(img_feat, self.proj_img_mlp)

        proj_navigable_feat = proj_masking(navigable_feat, self.proj_navigable_mlp, navigable_mask)

        weighted_img_feat, _ = self.soft_attn(self.h0_fc(h_0), proj_img_feat, img_feat)

        concat_input = torch.cat((pre_feat, weighted_img_feat), 1)

        h_1, c_1 = self.lstm(self.dropout(concat_input), (h_0, c_0))

        h_1_drop = self.dropout(h_1)

        # use attention on language instruction
        weighted_ctx, ctx_attn = self.soft_attn(self.h1_fc(h_1_drop), self.dropout(ctx), mask=ctx_mask)
        h_tilde = self.proj_out(weighted_ctx)

        logit = torch.bmm(proj_navigable_feat, h_tilde.unsqueeze(2)).squeeze(2)

        return h_1, c_1, weighted_ctx, weighted_ctx.clone(), ctx_attn, logit, navigable_mask, loss


class EnvDropFollower(nn.Module):
    ''' An unrolled LSTM with attention over instructions for decoding navigation actions. '''

    def __init__(self, img_feat_input_dim, rnn_hidden_size,
                       dropout_ratio=0.5, angle_feat_size=128, 
                       featdropout=0.5, max_navigable=None, embedding_size=64, **kwargs):
        super(EnvDropFollower, self).__init__()
        feature_size = img_feat_input_dim
        hidden_size  = rnn_hidden_size
        self.embedding_size = embedding_size #64
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.max_navigable = max_navigable
        self.angle_feat_size = angle_feat_size
        self.embedding = nn.Sequential(
            nn.Linear(angle_feat_size, self.embedding_size),
            nn.Tanh()
        )
        self.drop = nn.Dropout(p=dropout_ratio)
        self.drop_env = nn.Dropout(p=featdropout)
        self.lstm = nn.LSTMCell(embedding_size+feature_size, hidden_size)
        self.feat_att_layer = EnvDropSoftDotAttention(hidden_size, feature_size)
        self.attention_layer = EnvDropSoftDotAttention(hidden_size, hidden_size)
        self.candidate_att_layer = EnvDropSoftDotAttention(hidden_size, feature_size)

    def forward(self, feature, cand_feat, prev_feat, prev_h1, h_0, c_0, ctx, none,
                      navigable_index=None, ctx_mask=None):
        '''
        Takes a single step in the decoder LSTM (allowing sampling).
        action: batch x angle_feat_size
        feature: batch x 36 x (feature_size + angle_feat_size)
        cand_feat: batch x cand x (feature_size + angle_feat_size)
        h_0: batch x hidden_size
        prev_h1: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        already_dropfeat: used in EnvDrop
        '''
        batch_size = feature.shape[0]
        action = prev_feat[:,-self.angle_feat_size:]
        action_embeds = self.embedding(action)
        loss = 0

        # Adding Dropout
        action_embeds = self.drop(action_embeds)

        prev_h1_drop = self.drop(prev_h1)

        attn_feat, _ = self.feat_att_layer(prev_h1_drop, feature, output_tilde=False)

        concat_input = torch.cat((action_embeds, attn_feat), 1) # (batch, embedding_size+feature_size)
        h_1, c_1 = self.lstm(concat_input, (prev_h1, c_0))

        h_1_drop = self.drop(h_1)
        h_tilde, alpha = self.attention_layer(h_1_drop, ctx, (ctx_mask==0))

        # Adding Dropout
        h_tilde_drop = self.drop(h_tilde)

        _, logit = self.candidate_att_layer(h_tilde_drop, cand_feat, output_prob=False)

        index_length = [len(_index)+1 for _index in navigable_index]
        navigable_mask = create_mask(batch_size, self.max_navigable, index_length)

        return h_1, c_1, h_tilde, h_tilde, h_tilde, logit, navigable_mask, loss
