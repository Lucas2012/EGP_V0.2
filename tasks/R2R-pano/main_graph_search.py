import sys
import argparse
import numpy as np

import torch

from collections       import OrderedDict
from smna_models.env   import R2RBatch, ImageFeatures
from eval              import Evaluation
from utils             import save_checkpoint
from smna_models.utils import read_vocab, Tokenizer, vocab_pad_idx, timeSince, try_cuda, setup

from trainer_graph_search import GraphSeq2SeqTrainer
from agents               import GraphPanoSeq2SeqAgent
from models               import GraphPolicyNetwork


parser = argparse.ArgumentParser(description='PyTorch for Matterport3D Agent with Evolving Graphical Planner')
# General options
parser.add_argument('--exp_name', default='experiments_', type=str,
                    help='name of the experiment. \
                        It decides where to store samples and models')

# Training options
parser.add_argument('--train_vocab',
                    default='tasks/R2R-pano/data/train_vocab.txt',
                    type=str, help='path to training vocab')
parser.add_argument('--trainval_vocab',
                    default='tasks/R2R-pano/data/trainval_vocab.txt',
                    type=str, help='path to training and validation vocab')
parser.add_argument('--img_feat_dir',
                    default='img_features/ResNet-152-imagenet.tsv',
                    type=str, help='path to pre-cached image features')
parser.add_argument('--dataset_name',
                    default='R4R',
                    type=str, help="Name of dataset. ['R2R', 'R4R']")
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--batch_size', default=-1, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--train_iters_epoch', default=200, type=int,
                    help='number of iterations per epoch')
parser.add_argument('--max_num_epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--eval_every_epochs', default=1, type=int,
                    help='how often do we eval the trained model')
parser.add_argument('--seed', default=2049, type=int,
                    help='random seed')
parser.add_argument('--train_data_augmentation', default=0, type=int,
                    help='Training with the synthetic data generated with speaker')
parser.add_argument('--epochs_data_augmentation', default=5, type=int,
                    help='Number of epochs for training with data augmentation first')

# General model options
parser.add_argument('--arch', default='cogrounding', type=str,
		            help='options: cogrounding | speaker-baseline')
parser.add_argument('--max_navigable', default=16, type=int,
                    help='maximum number of navigable locations in the dataset is 15 \
                         we add one because the agent can decide to stay at its current location')
parser.add_argument('--use_ignore_index', default=1, type=int,
                    help='ignore target after agent has ended')

# Agent options
parser.add_argument('--follow_gt_traj', default=0, type=int,
                    help='the shortest path to the goal may not match with the instruction if we use student forcing, '
                         'we provide option that the next ground truth viewpoint will try to steer back to the original'
                         'ground truth trajectory')
parser.add_argument('--teleporting', default=1, type=int,
                    help='teleporting: jump directly to next viewpoint. If not enabled, rotate and forward until you reach the '
                         'viewpoint with roughly the same heading')
parser.add_argument('--max_episode_len', default=12, type=int,
                    help='maximum length of episode')
parser.add_argument('--feedback_training', default='sample2action', type=str,
                    help='options: sample | mistake (this is the feedback for training only)')
parser.add_argument('--feedback', default='argmax', type=str,
                    help='options: sample | argmax (this is the feedback for testing only)')
parser.add_argument('--fix_action_ended', default=1, type=int,
                    help='Action set to 0 if ended. This prevent the model keep getting loss from logit after ended')

# Image context
parser.add_argument('--img_feat_input_dim', default=2176, type=int,
                    help='ResNet-152: 2048, if use angle, the input is 2176')
parser.add_argument('--img_fc_dim', default=(128,), nargs="+", type=int)
parser.add_argument('--img_fc_use_batchnorm', default=1, type=int)
parser.add_argument('--mlp_share', default=1, type=int)
parser.add_argument('--img_dropout', default=0.5, type=float)
parser.add_argument('--mlp_relu', default=1, type=int, help='Use ReLu in MLP module')
parser.add_argument('--img_fc_use_angle', default=1, type=int,
                    help='add relative heading and elevation angle into image feature')

# Language model
parser.add_argument('--remove_punctuation', default=0, type=int,
                    help='the original ''encode_sentence'' does not remove punctuation'
                         'we provide an option here.')
parser.add_argument('--reversed', default=1, type=int,
                    help='option for reversing the sentence during encoding')
parser.add_argument('--lang_embed', default='lstm', type=str, help='options: lstm ')
parser.add_argument('--word_embedding_size', default=256, type=int,
                    help='default embedding_size for language encoder')
parser.add_argument('--rnn_hidden_size', default=256, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--rnn_num_layers', default=1, type=int)
parser.add_argument('--rnn_dropout', default=0.5, type=float)
parser.add_argument('--max_cap_length', default=80, type=int, help='maximum length of captions')

# Evaluation options
parser.add_argument('--eval_only', default=0, type=int,
                    help='No training. Resume from a model and run evaluation')
parser.add_argument('--eval_beam', default=0, type=int,
                    help='No training. Resume from a model and run with beam search')
parser.add_argument('--beam_size', default=5, type=int,
                    help='The number of beams used with beam search')

# Output options
parser.add_argument('--results_dir',
                    default='tasks/R2R-pano/results/',
                    type=str, help='where to save the output results for computing accuracy')
parser.add_argument('--log_dir',
                    default='tensorboard_logs/pano-seq2seq',
                    type=str, help='path to tensorboard log files')
parser.add_argument('--checkpoint_dir',
                    default='tasks/R2R-pano/checkpoints/pano-seq2seq/',
                    type=str, help='where to save trained models')
parser.add_argument('--vis_file_dir',
                    default='', #'vis_files/',
                    type=str, help='visualization files')

# GSA options
parser.add_argument('--training_mode',
                    default='GSA_Graph_supervised_pure_scratch',
                    type=str, help='training mode')
parser.add_argument('--joint_training',
                    default=0,
                    type=int, help='whether to use joint training')
parser.add_argument('--GSA_top_K',
                    default=20,
                    type=int, help='top K actions to be cached')
parser.add_argument('--actor_policy',
                    default='lstm',
                    type=str, help='none | lstm | mlp')
parser.add_argument('--base_model_lr_scale',
                    default=1,
                    type=float, help='scale the learning rate of base models')
parser.add_argument('--initial_sample_iters',
                    default=40,
                    type=int, help='number of iterations for collecting samples')
parser.add_argument('--train_actor_every_k',
                    default=1,
                    type=int, help='train actor network for every k iters')
parser.add_argument('--max_mp_steps',
                    default=3,
                    type=int, help='max number of message passing steps for graph based policy')
parser.add_argument('--no_h_feature',
                    default=0,
                    type=int, help='whether to use h features')
parser.add_argument('--graph_emb_dim',
                    default=64,
                    type=int, help='')
parser.add_argument('--graph_dim',
                    default=32,
                    type=int, help='')
parser.add_argument('--graph_teacher_option',
                    default='best_candidate',
                    type=str, help='')
parser.add_argument('--data_fraction',
                    default=1.0,
                    type=float, help='')
parser.add_argument('--separate_query',
                    default=1,
                    type=int, help='')
parser.add_argument('--use_smna_arch',
                    default=1,
                    type=int, help='')
parser.add_argument('--use_all_query',
                    default=0,
                    type=int, help='')
parser.add_argument('--use_cur_pos_h',
                    default=0,
                    type=int, help='')
parser.add_argument('--graph_attention',
                    default=0,
                    type=int, help='')
parser.add_argument('--gnn_dropout',
                    default=0,
                    type=float, help='')
parser.add_argument('--use_moving_window',
                    default=0,
                    type=int, help='')
parser.add_argument('--moving_window_size',
                    default=-1,
                    type=int, help='')
parser.add_argument('--ctx_attend',
                    default=0,
                    type=int, help='')
parser.add_argument('--graph_attention_type',
                    default='sigmoid',
                    type=str, help='sigmoid | sigmoid_single')
parser.add_argument('--use_pretraining',
                    default=0,
                    type=int, help='speaker follower data augmentation as pretraining step')
parser.add_argument('--pretraining_epochs',
                    default=-1,
                    type=int, help='')

# pooling options
parser.add_argument('--use_graph_pooling',
                    default=0,
                    type=int, help='')
parser.add_argument('--pooling_num_node',
                    default=-1,
                    type=int, help='')
parser.add_argument('--pooling_graph_dim',
                    default=-1,
                    type=int, help='')
parser.add_argument('--planner_graph_dim',
                    default=-1,
                    type=int, help='')
parser.add_argument('--pooling_mp_steps',
                    default=-1,
                    type=int, help='')
parser.add_argument('--normalize_pooling',
                    default='',
                    type=str, help='')
parser.add_argument('--pooling_channels',
                    default=-1,
                    type=int, help='')


def train_setup(args, train_splits=['train']):
    val_splits = ['val_seen', 'val_unseen']
    TRAIN_VOCAB = 'tasks/R2R-pano/data/train_FAST_vocab.txt'
    vocab = TRAIN_VOCAB

    if args.use_pretraining:
        train_env, val_envs, pretrain_env = make_env_and_models(
                          args, vocab, train_splits, val_splits)
        return train_env, val_envs, pretrain_env

    else:
        train_env, val_envs = make_env_and_models(
            args, vocab, train_splits, val_splits)
        return train_env, val_envs


def make_env_and_models(args, train_vocab_path, train_splits, test_splits):
    image_features_list = ImageFeatures.from_args(args)
    vocab = read_vocab(train_vocab_path)
    tok = Tokenizer(vocab=vocab)
    train_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=train_splits, tokenizer=tok, fraction=args.data_fraction,
                         dataset_name=args.dataset_name, opts=args) if len(train_splits) > 0 else None
    test_envs = OrderedDict([(
        split, (R2RBatch(image_features_list, batch_size=args.batch_size,
                         splits=[split], tokenizer=tok, fraction=1.0, 
                         dataset_name=args.dataset_name, opts=args),
                Evaluation([split], dataset_name=args.dataset_name)))
        for split in test_splits])

    if args.use_pretraining:
        pretrain_splits = ['literal_speaker_data_augmentation_paths']
        pretrain_env = R2RBatch(image_features_list, batch_size=args.batch_size,
                                splits=pretrain_splits, tokenizer=tok, fraction=args.data_fraction,
                                dataset_name=args.dataset_name, opts=args) if len(pretrain_splits) > 0 else None
        return train_env, test_envs, pretrain_env
    else:
        return train_env, test_envs


# Main experiment
def main(opts):

    # set manual_seed and build vocab
    setup(opts, opts.seed)

    opts.use_moving_window = 1 if opts.moving_window_size > 0 else 0

    # reset max_episode_len for R4R environment
    if opts.dataset_name == 'R4R':
        opts.max_episode_len   = 2 * opts.max_episode_len
        opts.train_iters_epoch = 2 * opts.train_iters_epoch
        if opts.use_moving_window:
            opts.batch_size        = 12
        else:
            opts.batch_size        = 12

    # set opts based on training mode
    print('Architecture: ', opts.arch)

    print(opts)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create a batch training environment that will also preprocess text
    opts.use_SF_angle = True
    opts.train_vocab = 'tasks/R2R-pano/data/train_FAST_vocab.txt'
    vocab = read_vocab(opts.train_vocab)
    from smna_models.utils import Tokenizer
    tok = Tokenizer(vocab)

    if 'scratch' in opts.training_mode:
        opts.training_mode = opts.training_mode[:-8]
        assert opts.joint_training == 1 

    from smna_models.make_follower import make_follower
    encoder, model = make_follower(opts, vocab, no_load=True)

    # create policy network
    if opts.training_mode in ['GSA_Graph_supervised_pure']:
        graph_pooling_config = [opts.use_graph_pooling, 
                                opts.pooling_num_node,
                                opts.pooling_graph_dim,
                                opts.planner_graph_dim,
                                opts.pooling_mp_steps,
                                opts.normalize_pooling,
                                opts.pooling_channels]
        if opts.mlp_share:
            visual_mlp_graph = model.visual_mlp
        else:
            visual_mlp_graph = None

        actor_network = GraphPolicyNetwork(policy_mapping=opts.actor_policy, \
                                           policy_indim=opts.rnn_hidden_size, \
                                           feature_dim=opts.img_feat_input_dim, \
                                           emb_dim=opts.graph_emb_dim, \
                                           graph_dim=opts.graph_dim, \
                                           node_history=True, \
                                           edge_history=False, \
                                           use_global=True, \
                                           use_attention=opts.graph_attention, \
                                           graph_attention_type=opts.graph_attention_type, \
                                           message_normalize=False, \
                                           no_h_feature=opts.no_h_feature, \
                                           feature_pool='max', \
                                           action_query_dim=2*opts.rnn_hidden_size, \
                                           visual_mlp=visual_mlp_graph, \
                                           use_all_query=opts.use_all_query, \
                                           use_cur_pos_h=opts.use_cur_pos_h, \
                                           graph_pooling_config=graph_pooling_config, \
                                           use_ctx_attend=opts.ctx_attend, \
                                           gnn_dropout=opts.gnn_dropout)

        actor_network = actor_network.to(device)
    else:
        actor_network = None

    # get optimizer list
    optimizers          = OrderedDict()
    agent_params        = list(encoder.parameters()) + list(model.parameters())
    optimizers['agent'] = torch.optim.Adam(agent_params, lr=opts.learning_rate)
    optimizers['policy_network'] = torch.optim.Adam(actor_network.parameters(), lr=opts.learning_rate / opts.base_model_lr_scale)

    # set up environments
    train_splits = ['train']
    if opts.use_pretraining:
        train_env, val_envs, pretrain_env = train_setup(opts)
        opts.max_num_epochs = opts.max_num_epochs + opts.pretraining_epochs
    else:
        train_env, val_envs = train_setup(opts)

    # create agent
    agent_kwargs = OrderedDict([
        ('opts', opts),
        ('env', train_env),
        ('results_path', ""),
        ('encoder', encoder),
        ('model', model),
        ('actor_network', actor_network),
        ('feedback', opts.feedback),
    ])
    agent   = GraphPanoSeq2SeqAgent(**agent_kwargs)

    # setup trainer
    trainer = GraphSeq2SeqTrainer(opts, agent, optimizers, opts.train_iters_epoch)

    # start training
    best_success_rate_env =  0.0
    for epoch in range(opts.start_epoch, opts.max_num_epochs + 1):

        if opts.use_pretraining and epoch < opts.pretraining_epochs:
            trainer.train(epoch, pretrain_env)
        else:
            trainer.train(epoch, train_env)

        if epoch % opts.eval_every_epochs == 0:
            success_rate_env    = []
            for val_env in val_envs.items():
                success_rate_env.append(trainer.eval(epoch, val_env))
            success_rate_compare_env    = success_rate_env[1]

            is_best_env = None
            # remember best val_seen success rate and save checkpoint
            if len(success_rate_env) > 0:
                is_best_env = success_rate_compare_env >= best_success_rate_env
                best_success_rate_env = max(success_rate_compare_env, best_success_rate_env)
                print("--> Highest val_unseen success rate env: {}".format(best_success_rate_env))

            # save the model if it is the best so far
            if opts.use_pretraining and epoch == opts.pretraining_epochs - 1:
                aug_appendix = '_aug'
            else:
                aug_appendix = ''
            save_checkpoint(OrderedDict([
                ('opts', opts),
                ('epoch', epoch + 1),
                ('state_dict', model.state_dict()),
                ('encoder_state_dict', encoder.state_dict()),
                ('actor_state_dict', actor_network.state_dict() if actor_network is not None else None),
                ('best_success_rate_env', best_success_rate_env),
                ('max_episode_len', opts.max_episode_len),
            ]), is_best_env == True, checkpoint_dir=opts.checkpoint_dir, name=opts.exp_name + aug_appendix)

        # save the model if it is the best so far
        if opts.use_pretraining and epoch == opts.pretraining_epochs - 1:
            aug_appendix = '_aug'
            save_checkpoint(OrderedDict([
                ('opts', opts),
                ('epoch', epoch + 1),
                ('state_dict', model.state_dict()),
                ('encoder_state_dict', encoder.state_dict()),
                ('actor_state_dict', actor_network.state_dict() if actor_network is not None else None),
                ('best_success_rate_env', best_success_rate_env),
                ('max_episode_len', opts.max_episode_len),
            ]), False, checkpoint_dir=opts.checkpoint_dir, name=opts.exp_name + aug_appendix)


    print("--> Finished training")


if __name__ == '__main__':
    ImageFeatures.add_args(parser)
    opts = parser.parse_args()
    main(opts)
