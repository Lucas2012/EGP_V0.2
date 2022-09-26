import os
import sys
import re
import string
import json
import time
import math
import shutil
import warnings
from collections import Counter
import numpy as np
import networkx as nx
from collections import OrderedDict
# from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.nn.functional import softmax as softmax
from torch.autograd import Variable

base_vocab = ['<PAD>', '<START>', '<EOS>', '<UNK>']
padding_idx = base_vocab.index('<PAD>')
end_token_idx = base_vocab.index('<EOS>')
period_idx = 5 # this is hardcoded right now for jank implementation purposes.

path_len_prob_random_train_R2R = [30, 30, 30, 60, 60, 60, 10] #### THIS IS AN ARBITRARY VECTOR. SHOULD BE CHANGED.
path_len_prob_random_val_R2R   = [30, 30, 30, 60, 60, 60, 10] #### THIS IS AN ARBITRARY VECTOR. SHOULD BE CHANGED.
path_len_prob_random_train_R4R = [30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 90, 90, 60, 60, 60, 60, 30] #### THIS IS AN ARBITRARY VECTOR. SHOULD BE CHANGED.
path_len_prob_random_val_R4R   = [30, 30, 30, 30, 30, 30, 30, 60, 60, 60, 90, 90, 60, 60, 60, 60, 30] #### THIS IS AN ARBITRARY VECTOR. SHOULD BE CHANGED.

path_len_prob_R2R = [0, 0, 8, 1655, 1325, 1687, 0]
path_len_prob_R4R = [0, 0, 0, 0, 0, 0, 0, 5, 593, 2981, 5369, 7083, 5804, 3184, 802, 89, 1]

class Entropy(nn.Module):
    def __init__(self, size_average=True):
        super(Entropy, self).__init__()
        self.size_average = size_average

    def forward(self, logit_p):
        dims = logit_p.dim()
        p = softmax(logit_p, dim=dims-1)
        p = p.view(-1, p.size(-1))
        entropy = - p * torch.log(p + 1e-6)
        entropy = entropy.sum()
        if self.size_average:
          entropy = entropy / p.size(0) / p.size(1)
        return entropy

class SoftBCELoss(nn.Module):
    def __init__(self):
        super(SoftBCELoss, self).__init__()

    def forward(self, p, soft_label):
        p = torch.clamp(p, min=1e-3, max=(1-1e-3))
        soft_loss  = -(soft_label * torch.log(p))
        soft_loss += -((1 - soft_label) * torch.log(1 - p))
        soft_loss  = soft_loss.mean()
        return soft_loss

class CrossEntropy(nn.Module):
    def __init__(self, size_average=True):
        super(CrossEntropy, self).__init__()
        self.size_average = size_average

    def forward(self, logit_p, logit_q, probability=False, seq_lengths=None):
        dims = logit_p.dim()
        # Logit is set up such that the 1st dimension always contains unnormalized logits
        q = softmax(logit_q, dim=1)
        if not probability:
          p = softmax(logit_p, dim=1)
        else:
          p = logit_p
        cross_entropy = -p * torch.log(q + 1e-6)
        if seq_lengths is not None:
          seq_lengths = torch.Tensor(seq_lengths).cuda().float()
          mask = torch.arange(logit_p.shape[-1]).cuda().unsqueeze(0).expand(seq_lengths.shape[0], logit_p.shape[-1])
          mask = (mask.float() < seq_lengths.unsqueeze(1)).float()
          mask = mask.unsqueeze(1).expand(logit_p.shape)
          cross_entropy = cross_entropy * mask
        if self.size_average:
          if seq_lengths is not None:
            cross_entropy = cross_entropy.sum() / (mask.sum() + 1)
          else:
            cross_entropy = cross_entropy.mean()
        else:
          cross_entropy = cross_entropy.sum()
        return cross_entropy


def setup(opts, seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Check for vocabs
    if not os.path.exists(opts.train_vocab):
        write_vocab(build_vocab(splits=['train'], dataset_name=opts.dataset_name), opts.train_vocab)
    if not os.path.exists(opts.trainval_vocab):
        write_vocab(build_vocab(splits=['train', 'val_seen', 'val_unseen'], dataset_name=opts.dataset_name), opts.trainval_vocab)


def load_nav_graphs(scans):
    """ Load connectivity graph for each scan """

    def distance(pose1, pose2):
        """ Euclidean distance between two graph poses """
        return ((pose1['pose'][3] - pose2['pose'][3]) ** 2
                + (pose1['pose'][7] - pose2['pose'][7]) ** 2
                + (pose1['pose'][11] - pose2['pose'][11]) ** 2) ** 0.5

    graphs = OrderedDict()
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = OrderedDict()
            data = json.load(f)
            for i, item in enumerate(data):
                if item['included']:
                    for j, conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_datasets(splits, opts=None, dataset_name='R2R'):
    if dataset_name not in ['R2R', 'R4R']:
        raise ValueError("Dataset name must be either 'R2R' or 'R4R'")
    prefix='tasks/R2R-pano/data/{}'.format(dataset_name)
    data = []
    for split in splits:
        assert split in ['train', 'val_seen', 'val_unseen', 'val_unseen_full', 'test', 'train_val_seen', 'synthetic', 'literal_speaker_data_augmentation_paths']
        if split == 'synthetic':
            if dataset_name == 'R4R':
                raise ValueError("No synthetic dataset for R4R benchmark.")
            with open('tasks/R2R-pano/data/R2R_literal_speaker_data_augmentation_paths.json') as f:
                data += json.load(f)
        else:
            with open(prefix + '_%s.json' % split) as f:
                data += json.load(f)

    max_len = 0
    length_dict = {}
    for ele in data:
        length = len(ele['path'])
        if length > max_len:
            max_len = length 
        if length in length_dict.keys():
            length_dict[length] += 1
        else:
            length_dict[length] = 0

    length_list = [0] * max_len
    for i in range(len(length_list)):
        if i+1 in length_dict.keys():
            length_list[i] = length_dict[i+1]
    print('Dataset name - length: ', dataset_name, length_list)

    return data


class Tokenizer(object):
    """ Class to tokenize and encode a sentence. """
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')  # Split on any non-alphanumeric character

    def __init__(self, remove_punctuation=False, reversed=True, vocab=None, encoding_length=20):
        self.remove_punctuation = remove_punctuation
        self.reversed = reversed
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.word_to_index = {}
        # self.client = CoreNLPClient(default_annotators=['ssplit', 'tokenize', 'pos'])
        if vocab:
            for i, word in enumerate(vocab):
                self.word_to_index[word] = i

    def split_sentence(self, sentence):
        """ Break sentence into a list of words and punctuation """
        toks = []
        for word in [s.strip().lower() for s in self.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if
                     len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def encode_sentence(self, sentence):
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')
        encoding = []

        splited = self.split_sentence(sentence)
        if self.reversed:
            splited = splited[::-1]

        if self.remove_punctuation:
            splited = [word for word in splited if word not in string.punctuation]

        for word in splited:  # reverse input sentences
            if word in self.word_to_index:
                encoding.append(self.word_to_index[word])
            else:
                encoding.append(self.word_to_index['<UNK>'])

        encoding.append(self.word_to_index['<EOS>'])
        encoding.insert(0, self.word_to_index['<START>'])

        if len(encoding) < self.encoding_length:
            encoding += [self.word_to_index['<PAD>']] * (self.encoding_length - len(encoding))
        return np.array(encoding[:self.encoding_length])

    def decode_sentence(self, encoding):
        sentence = []
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.vocab[ix])
        if self.reversed:
            sentence = sentence[::-1]
        return " ".join(sentence)


def build_vocab(splits=['train'], min_count=5, start_vocab=base_vocab, dataset_name='R2R'):
    """ Build a vocab, starting with base vocab containing a few useful tokens. """
    count = Counter()
    t = Tokenizer()
    data = load_datasets(splits, dataset_name=dataset_name)
    for item in data:
        for instr in item['instructions']:
            count.update(t.split_sentence(instr))
    vocab = list(start_vocab)
    for word, num in count.most_common():
        if num >= min_count:
            vocab.append(word)
        else:
            break
    return vocab


def write_vocab(vocab, path):
    print('Writing vocab of size %d to %s' % (len(vocab), path))
    with open(path, 'w') as f:
        for word in vocab:
            f.write("%s\n" % word)


def read_vocab(path):
    vocab = []
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def set_tb_logger(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        import shutil
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints/', name='checkpoint'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = checkpoint_dir + name + '.pth.tar'
    torch.save(state, filename)
    if is_best:
        best_filename = checkpoint_dir + name + '_model_best.pth.tar'
        shutil.copyfile(filename, best_filename)


def is_experiment():
    """
    A small function for developing on MacOS. When developing, the code will not load the full dataset
    """
    if sys.platform != 'darwin':
        return True
    else:
        return False


def resume_training(opts, model, encoder, translator, speaker, optimizers):
    if opts.resume == 'latest':
        file_extension = 'pre_train_models.pth.tar'
    elif opts.resume == 'best':
        file_extension = 'pre_train_models_model_best.pth.tar'
    else:
        raise ValueError('Unknown resume option: {}'.format(opts.resume))
    #opts.resume = opts.checkpoint_dir + opts.exp_name + file_extention
    if opts.ssl_beta == 0.0:
        which_pretraining = 'trans'
    else:
        which_pretraining = 'joint'
    neur_params = 'neurlen{}_neursize{}'.format(opts.neuralese_len, opts.neuralese_vocab_size)
    opts.resume = os.path.join(opts.resume_base_dir, 
                               opts.dataset_name,
                               neur_params,
                               which_pretraining,
                               opts.sampling_strategy,
                               file_extension)
                                
    if os.path.isfile(opts.resume):
        if is_experiment():
            checkpoint = torch.load(opts.resume)
        else:
            checkpoint = torch.load(opts.resume, map_location=lambda storage, loc: storage)
        #opts.start_epoch = checkpoint['epoch']
        try:
            opts.max_episode_len = checkpoint['max_episode_len']
        except:
            pass
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded in agent parameters')
        if encoder is not None:
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print('Loaded in encoder parameters.')
        if translator is not None and checkpoint['translator_state_dict'] is not None:
            translator.load_state_dict(checkpoint['translator_state_dict'])
            print('Loaded in translator parameters.')
        if speaker is not None and checkpoint['speaker_state_dict'] is not None:
            speaker.load_state_dict(checkpoint['speaker_state_dict'])
            print('Loaded in speaker parameters.')
        try:
            loaded_optimizers = checkpoint['optimizers']
            for optimizer_name in optimizers.keys():
                if optimizer_name in loaded_optimizers.keys():
                    optimizers[optimizer_name].load_state_dict(loaded_optimizers[optimizer_name].state_dict())
                    print('Loading in optimizer: {}'.format(optimizer_name))
        except:
            print('No optimizers found to load.')
            pass 
        try:
            best_loader = checkpoint['best_success_rate_loader']
        except:
            best_loader = 0.0
        try:
            best_env = checkpoint['best_success_rate_env']
        except:
            best_env = 0.0
        print("=> loaded checkpoint '{}'"# (epoch {})"
              .format(opts.resume))#, checkpoint['epoch']))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(opts.resume))
    return model, encoder, translator, speaker, optimizers, best_loader, best_env

def pad_tensor(tensor, length):
    """Pad a tensor, given by the max length"""
    if tensor.size(0) == length:
        return tensor
    return torch.cat([tensor, tensor.new(length - tensor.size(0),
                                  *tensor.size()[1:]).zero_()])

def find_length(list_tensors):
    """find the length of list of tensors"""
    if type(list_tensors[0]) is np.ndarray:
        length = [x.shape[0] for x in list_tensors]
    else:
        length = [x.size(0) for x in list_tensors]
    return length

def pad_list_tensors(list_tensor, max_length=None):
    """Pad a list of tensors and return a list of tensors"""
    tensor_length = find_length(list_tensor)

    if max_length is None:
        max_length = max(tensor_length)

    list_padded_tensor = []
    for tensor in list_tensor:
        if tensor.size(0) != max_length:
            tensor = pad_tensor(tensor, max_length)
        list_padded_tensor.append(tensor)

    return torch.stack(list_padded_tensor), tensor_length


def calculate_pis(logits):
    #unnormed_pis = logits.exp() # exponentiate to invert log
    #pis = unnormed_pis / unnormed_pis.sum(-1).unsqueeze(-1) # normalize to sum to 1
    pis = softmax(logits, dim=-1)
    return pis   


def kl_div(p, q):
    return torch.sum(p) - torch.sum(q)
