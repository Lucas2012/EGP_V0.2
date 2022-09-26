from SF_utils.SF_model import SpeakerEncoderLSTM, SpeakerDecoderLSTM
from SF_utils.SF_speaker import Seq2SeqSpeaker
from SF_utils.SF_utils import read_vocab, Tokenizer, try_cuda

import numpy as np

MAX_INSTRUCTION_LENGTH = 80

word_embedding_size = 300
glove_path = 'tasks/R2R-pano/SF_utils/data/train_glove.npy'
action_embedding_size = 2048+128
hidden_size = 512
bidirectional = False
dropout_ratio = 0.5
FEATURE_SIZE = 2048+128


def make_env_and_models(env, vocab_path, test_instruction_limit=None):
    vocab = read_vocab(vocab_path)
    tok = Tokenizer(vocab=vocab)
    env.tokenizer = tok

    enc_hidden_size = hidden_size//2 if bidirectional else hidden_size
    glove = np.load(glove_path)
    feature_size = FEATURE_SIZE
    encoder = try_cuda(SpeakerEncoderLSTM(
        action_embedding_size, feature_size, enc_hidden_size, dropout_ratio,
        bidirectional=bidirectional))
    decoder = try_cuda(SpeakerDecoderLSTM(
        len(vocab), word_embedding_size, hidden_size, dropout_ratio,
        glove=glove))

    return env, encoder, decoder


def selfplay_speaker_setup(env):
    SUBTRAIN_VOCAB = 'tasks/R2R-pano/SF_utils/data/sub_train_vocab.txt'
    TRAIN_VOCAB    = 'tasks/R2R-pano/SF_utils/data/train_vocab.txt'
    TRAINVAL_VOCAB = 'tasks/R2R-pano/SF_utils/data/trainval_vocab.txt'

    vocab = TRAIN_VOCAB

    env, encoder, decoder = make_env_and_models(env, vocab, test_instruction_limit=1)
    agent = Seq2SeqSpeaker(env, "", encoder, decoder, MAX_INSTRUCTION_LENGTH)

    return agent, env


def entry_point(env, speaker_model_prefix):
    speaker, env = selfplay_speaker_setup(env)
    speaker.load(speaker_model_prefix)
    return speaker, env
