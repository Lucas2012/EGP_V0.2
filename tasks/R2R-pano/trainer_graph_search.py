import sys
import time
import math
import numpy as np

from collections import OrderedDict
import torch
from utils import AverageMeter, load_datasets

import pickle


# This is the code for Reinforced Graph Search Algorithm
class GraphSeq2SeqTrainer():
    """Trainer for training and validation process"""
    def __init__(self, opts, agent, optimizers, train_iters_epoch=100):
        self.opts = opts
        self.agent = agent
        self.optimizers = optimizers
        self.train_iters_epoch = train_iters_epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def backward_zero_grad(self, optimizers, keys=None):
        if keys is None:
          keys = optimizers.keys()
        for k in keys:
          optimizers[k].zero_grad()

    def backward_compute(self, optimizers, loss, keys=None):
        if keys is None:
          keys = optimizers.keys()
        loss.backward()
        for k in keys:
          optimizers[k].step()

    def train(self, epoch, train_env):
        batch_time = AverageMeter()
        losses = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()
        val_losses = AverageMeter()
        a_losses = AverageMeter()
        q_losses = AverageMeter()
        ent_losses = AverageMeter()
        val_acces = AverageMeter()

        print('Training on {} env ...'.format(train_env.splits[0]))
        # switch to train mode
        self.agent.env = train_env
        self.agent.encoder.train()
        self.agent.model.train()
        if 'GSA' in self.opts.training_mode:
            self.agent.actor_network.train()

        self.agent.feedback = self.opts.feedback_training
        self.agent.val_acc = None

        # load dataset path for computing ground truth distance
        self.agent.gt = OrderedDict()
        for item in load_datasets(train_env.splits, self.opts, dataset_name=self.opts.dataset_name):
            self.agent.gt[item['path_id']] = item

        end = time.time()

        for iter in range(1, self.train_iters_epoch + 1):
            a_loss = torch.zeros(1).to(self.device)
            q_loss = torch.zeros(1).to(self.device)
            ent_loss = torch.zeros(1).to(self.device)
            # rollout the agent
            if self.opts.training_mode == 'smna':
                if self.opts.arch == 'cogrounding':
                    loss, traj = self.agent.rollout_smna()
                else:
                    raise NotImplementedError()
            elif self.opts.training_mode == 'GSA_supervised' or self.opts.training_mode == 'GSA_Graph_supervised' \
                 or self.opts.training_mode == 'GSA_Graph_supervised_pure':
                self.agent.actor_network.feedback = self.opts.feedback_training
                batch_queue, traj, GSA_traj, actor_entropy, _, _, _, \
                    l2_reg_sum = self.agent.rollout_GSA_sac(is_train=True, return_l2_reg_sum=True)

                loss = self.agent.get_supervised_loss(GSA_traj)

            dist_from_goal = np.mean(self.agent.dist_from_goal)
            movement = np.mean(self.agent.traj_length)

            losses.update(loss.item(), self.opts.batch_size)
            a_losses.update(a_loss.item(), self.opts.batch_size)
            q_losses.update(q_loss.item(), self.opts.batch_size)
            ent_losses.update(ent_loss.item(), self.opts.batch_size)
            dists.update(dist_from_goal, self.opts.batch_size)
            movements.update(movement, self.opts.batch_size)

            if self.agent.val_acc is not None:
                val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

            optim_keys = None
            self.backward_zero_grad(self.optimizers, keys=optim_keys)
            self.backward_compute(self.optimizers, loss, keys=optim_keys)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, iter, self.train_iters_epoch, batch_time=batch_time,
                loss=losses))

            sys.stdout.flush()


    def eval(self, epoch, val_env):
        batch_time = AverageMeter()
        losses = AverageMeter()
        dists = AverageMeter()
        movements = AverageMeter()
        val_losses = AverageMeter()
        a_losses = AverageMeter()
        q_losses = AverageMeter()
        val_acces = AverageMeter()

        env_name, (env, evaluator) = val_env

        print('Evaluating on {} env ...'.format(env_name))

        self.agent.env = env
        self.agent.env.reset_epoch()
        self.agent.model.eval()
        self.agent.encoder.eval()
        if 'GSA' in self.opts.training_mode:
            self.agent.actor_network.eval()

        self.agent.feedback = self.opts.feedback
        self.agent.val_acc = None

        # load dataset path for computing ground truth distance
        self.agent.gt = OrderedDict()
        for item in load_datasets([env_name], dataset_name=self.opts.dataset_name):
            self.agent.gt[item['path_id']] = item
        val_iters_epoch = math.ceil(len(env.data) / self.opts.batch_size)
        self.agent.results = OrderedDict()
        looped = False
        iter = 1
        pickle_iter = 0

        with torch.no_grad():
            end = time.time()
            a_loss = torch.zeros(1).to(self.device)
            q_loss = torch.zeros(1).to(self.device)
            while True:

                if self.opts.eval_beam:
                    raise ValueError('We do not use beam search')
                else:
                    # rollout the agent
                    if self.opts.training_mode == 'smna':
                        if self.opts.arch == 'cogrounding':
                            loss, traj = self.agent.rollout_smna()
                        else:
                            raise NotImplementedError()
                    elif self.opts.training_mode == 'GSA_Graph_supervised_pure':
                        self.agent.actor_network.feedback = 'argmax'
                        batch_queue, traj, GSA_traj, actor_entropy, \
                            _, _, _, GSA_vis_traj = self.agent.rollout_GSA_sac(is_train=False, return_vis=True)
                        loss = self.agent.get_supervised_loss(GSA_traj)

                    else:
                        raise NotImplementedError()

                    dist_from_goal = np.mean(self.agent.dist_from_goal)
                    movement = np.mean(self.agent.traj_length)

                    losses.update(loss.item(), self.opts.batch_size)
                    a_losses.update(a_loss.item(), self.opts.batch_size)
                    q_losses.update(q_loss.item(), self.opts.batch_size)
                    dists.update(dist_from_goal, self.opts.batch_size)
                    movements.update(movement, self.opts.batch_size)
                    if self.agent.val_acc is not None:
                        val_acces.update(np.mean(self.agent.val_acc), self.opts.batch_size)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if len(self.opts.vis_file_dir) > 0:
                    for _, GSA_vis_traj_ele in enumerate(GSA_vis_traj):
                        pickle_file = self.opts.vis_file_dir + env_name + '_' +  str(pickle_iter) + '.p'
                        pickle.dump(GSA_vis_traj_ele, open(pickle_file, "wb" ))
                        pickle_iter += 1

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, iter, val_iters_epoch, batch_time=batch_time,
                    loss=losses, a_loss=a_losses, q_loss=q_losses))

                sys.stdout.flush()

                # write into results
                for traj_ in traj:
                    if traj_['instr_id'] in self.agent.results:
                        looped = True
                    else:
                        result = OrderedDict([
                            ('path', traj_['path']),
                            ('path_id', None),
                        ])
                        self.agent.results[traj_['instr_id']] = result
                if looped:
                    break
                iter += 1

        # dump into JSON file
        if self.opts.eval_beam:
            self.agent.results_path = '{}{}-beam_{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                             self.opts.beam_size, env_name, epoch)
        else:
            self.agent.results_path = '{}{}_{}_epoch_{}.json'.format(self.opts.results_dir, self.opts.exp_name,
                                                                     env_name, epoch)
        self.agent.write_results()
        score_summary, _ = evaluator.score(self.agent.results_path)
        result_str = ''
        success_rate = 0.0
        for metric, val in score_summary.items():
            result_str += '| {}: {} '.format(metric, val)
            if metric in ['success_rate']:
                success_rate = val

        print(result_str)

        return success_rate
