import torch
from torch import optim

import os
import os.path
import time
import numpy as np
import pandas as pd
import argparse

#import utils
from smna_models.model import TransformerEncoder, EncoderLSTM, AttnDecoderLSTM, CogroundDecoderLSTM, ProgressMonitor, DeviationMonitor
from smna_models.model import SpeakerEncoderLSTM, DotScorer
from smna_models.follower import Seq2SeqAgent
from smna_models.utils import read_vocab, try_cuda, vocab_pad_idx


def make_follower(args, vocab, no_load=False):
    enc_hidden_size = args.rnn_hidden_size//2 if args.bidirectional else args.rnn_hidden_size
    # glove = np.load(glove_path) if args.use_glove else None
    glove = None
    feature_size = args.img_feat_input_dim #FEATURE_SIZE
    # Encoder = TransformerEncoder if args.transformer else EncoderLSTM
    Encoder = EncoderLSTM
    Decoder = CogroundDecoderLSTM # if args.coground else AttnDecoderLSTM
    word_embedding_size = 256 # if args.coground else 300
    encoder = try_cuda(Encoder(
        len(vocab), word_embedding_size, enc_hidden_size, vocab_pad_idx,
        args.rnn_dropout, bidirectional=args.bidirectional, glove=glove))
    decoder = try_cuda(Decoder(
        args.img_feat_input_dim, args.rnn_hidden_size, args.rnn_dropout,
        feature_size=feature_size, num_head=-1))

    return encoder, decoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--img_feat_input_dim', default=2176, type=int,
                    help='ResNet-152: 2048, if use angle, the input is 2176')
    parser.add_argument('--rnn_hidden_size', default=512, type=int,
                    help='hidden size for the models')
    parser.add_argument('--bidirectional', default=0, type=int)
    parser.add_argument('--prog_monitor', default=0, type=int)
    parser.add_argument('--train_vocab', default='data/train_FAST_vocab.txt', type=str)
    parser.add_argument('--rnn_dropout', default=0.5, type=float)
    parser.add_argument('--num_head', default='none', type=str)
    
    args = parser.parse_args()
    vocab = read_vocab(args.train_vocab)
    encoder, decoder, prog_monitor = make_follower(args, vocab)
