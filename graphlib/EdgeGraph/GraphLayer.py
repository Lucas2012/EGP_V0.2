import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pdb

from graphlib.EdgeGraph.functions.EdgeFunction   import EdgeFunction
from graphlib.EdgeGraph.functions.NodeFunction   import NodeFunction
from graphlib.EdgeGraph.functions.GlobalFunction import GlobalFunction

class GraphLayer(nn.Module):
  '''
    Edge memory based graph layer.

    Input:  (U, {h_v}, {h_e}), (V,E)
    Output: (U, {h_v}, {h_e})
  '''
  def __init__(self, nodedims, edgedims, globaldims,
               residual=False, node_history=True, edge_history=False, \
               use_global=True, num_functions=2, rnn=False, \
               use_attention=False, message_normalize=False, \
               attention_type='sigmoid', dropout=0):
    super(GraphLayer, self).__init__()

    self.nodedim_in   = nodedims[0]
    self.nodedim_out  = nodedims[1]
    self.edgedim_in   = edgedims[0]
    self.edgedim_out  = edgedims[1]
    self.globaldim_in  = globaldims[0]
    self.globaldim_out = globaldims[1]
    self.use_global   = use_global

    ''' message updates functions '''
    # edge updates
    self.edgeUpdate    = EdgeFunction(self.nodedim_in, [self.edgedim_in, self.edgedim_out], \
                                      self.globaldim_in, residual=residual, history=edge_history, \
                                      use_global=use_global, num_functions=num_functions, use_attention=use_attention, \
                                      attention_type=attention_type, dropout=dropout)
    
    # node updates
    self.nodeUpdate    = NodeFunction([self.nodedim_in, self.nodedim_out], self.edgedim_out, \
                                       self.globaldim_in, residual=residual, history=node_history, \
                                       use_global=use_global, rnn=rnn, message_normalize=message_normalize, \
                                       dropout=dropout)

    if use_global:
      # global updates
      self.globalUpdate  = GlobalFunction(self.nodedim_out, self.edgedim_out, [self.globaldim_in, self.globaldim_out], residual=residual)

  def forward(self, graph_states, graph, extra_edge_feat=None):
    '''
      * K graphs per forward, all graphs share the same name space for node and edge

      input: 
         global state -> u:   [K, globaldim]
         node   state -> h_v: [K, |V|, nodedim]
         edge   state -> h_e: [K, |V|, |V|, edgedim]

         V -> nodes mask: [K, |V|]
         E -> edges mask: [K, |V|, |V|]

      output: 
         global state -> u:   [K, globaldim]
         node   state -> h_v: [K, |V|, nodedim]
         edge   state -> h_e: [K, |V|, |V|, edgedim]
    '''

    # inputs
    [u, h_v, h_e] = graph_states

    [V, E]        = graph
    # edge updates
    h_e = self.edgeUpdate(u, h_v, h_e, V, E, extra_edge_feat)
    
    # node updates
    h_v = self.nodeUpdate(u, h_v, h_e, V, E)
    
    if self.use_global:
      # global updates
      u   = self.globalUpdate(u, h_v, h_e, V, E)

    return u, h_v, h_e

if __name__ == '__main__':
  # test case

  numNode   = 10
  graphNum  = 24
  nodeDim   = 13
  edgeDim   = 15
  globalDim = 24

  graphLayer = GraphLayer([nodeDim, nodeDim+1], [edgeDim, edgeDim+1], [globalDim, globalDim+1])
  graphLayer.cuda()

  h_v = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  h_e = Variable(torch.zeros(graphNum, numNode, numNode, edgeDim)).cuda()
  u   = Variable(torch.zeros(graphNum, globalDim)).cuda()
  V   = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  E   = Variable(torch.LongTensor(graphNum, numNode, numNode).random_(0,2).float()).cuda()

  h_e = graphLayer([u, h_v, h_e], [V, E])
