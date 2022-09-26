import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pdb

class EdgeFunction(nn.Module):
  '''
    Input:  (U, {h_v}, {h_e}), (V,E)
    Output: (U, {h_v}, {h_e})
  '''
  def __init__(
          self,
          nodedim,
          edgedims,
          globaldim,
          residual,
          history=True,
          use_global=True,
          num_functions=2,
          use_attention=False,
          attention_type='sigmoid',
          dropout=0
    ):
    super(EdgeFunction, self).__init__()

    self.nodedim_in   = nodedim
    self.edgedim_in   = edgedims[0]
    self.edgedim_out  = edgedims[1]
    self.globaldim_in = globaldim
    self.residual     = residual
    self.history      = history
    self.use_global   = use_global
    self.num_functions = num_functions
    self.attention_type = attention_type

    ''' message updates function for edges '''

    self.edgeInfoDim = self.nodedim_in * 2
    if use_global:
      self.edgeInfoDim += self.globaldim_in
    if history:
      self.edgeInfoDim += self.edgedim_in

    self.function = []
    if use_attention:
      self.gates = []
    self.use_attention = use_attention
    self.edge_feat_map = nn.Sequential(
                           nn.Linear(self.nodedim_in, self.nodedim_in),
                           nn.ELU(),
                           nn.Linear(self.nodedim_in, self.edgedim_out)
                         )
    for i in range(0, num_functions):
      self.function  += [nn.Sequential(
                           nn.Linear(self.edgeInfoDim, self.edgeInfoDim),
                           nn.ELU(),
                           nn.Linear(self.edgeInfoDim, self.edgedim_out)
                         )]
      if use_attention:
        if self.attention_type == 'sigmoid':
          self.gates     += [nn.Sequential(
                               nn.Linear(self.edgeInfoDim, self.edgeInfoDim),
                               nn.ELU(),
                               nn.Linear(self.edgeInfoDim, self.edgedim_out),
                               nn.Sigmoid()
                             )]
        elif self.attention_type == 'sigmoid_single':
          self.gates     += [nn.Sequential(
                               nn.Linear(self.edgeInfoDim, self.edgeInfoDim),
                               nn.ELU(),
                               nn.Linear(self.edgeInfoDim, 1),
                               nn.Sigmoid()
                             )]
    self.function = nn.ModuleList(self.function)
    if use_attention:
      self.gates = nn.ModuleList(self.gates)

  def forward(self, u, h_v, h_e, V, E, extra_edge_feat=None):
    '''
      * K graphs per forward, all graphs share the same name space for node and edge

      input: 
         global state -> u:   [K, globaldim]
         node   state -> h_v: [K, |V|, nodedim]
         edge   state -> h_e: [K, |V|, |V|, edgedim]

         V -> nodes mask: [K, |V|]
         E -> edges mask: [K, |V|, |V|]

      output: 
         edge   state -> h_e: [K, |V|, |V|, edgedim]
    '''

    graphNum, numNode = h_v.size(0), h_v.size(1)

    vGridSize   = [graphNum, numNode, numNode]
    if V is not None:
      h_v = h_v * V.unsqueeze(-1).expand(graphNum, numNode, h_v.shape[-1])
    srcNodeGrid = h_v.unsqueeze(2).expand(vGridSize + [self.nodedim_in])
    tgtNodeGrid = h_v.unsqueeze(1).expand(vGridSize + [self.nodedim_in])
    info_vec = [srcNodeGrid, tgtNodeGrid]
    if self.use_global:
      globalNodeGrid = u.unsqueeze(1).unsqueeze(1).expand(vGridSize + [self.globaldim_in])
      info_vec += [globalNodeGrid]
    if self.history:
      info_vec += [h_e]
    edgeFeature = torch.cat(info_vec, dim=3) 

    E_size     = [E.size(0)] + [self.edgedim_out] + list(E.size()[1:3]) + [self.num_functions]
    E_expand   = E.unsqueeze(1).expand(E_size).cuda()
    h_e_update = torch.zeros(graphNum, numNode, numNode, self.edgedim_out).cuda()
    for i in range(1, self.num_functions): 
      h_e_update_i = self.function[i](edgeFeature[E[:,:,:,i]==1])
      if extra_edge_feat is not None:
        h_e_feat_i   = self.edge_feat_map(extra_edge_feat[E[:,:,:,i]==1])
        h_e_update_i += h_e_feat_i
      if self.use_attention:
        if extra_edge_feat is not None:
          #h_e_update_i = h_e_update_i * self.gates[i](edgeFeature[E[:,:,:,i]==1] + extra_edge_feat[E[:,:,:,i]==1])
          h_e_update_i = h_e_update_i * self.gates[i](edgeFeature[E[:,:,:,i]==1])
        else:
          h_e_update_i = h_e_update_i * self.gates[i](edgeFeature[E[:,:,:,i]==1])
      h_e_update[E[:,:,:,i]==1] = h_e_update_i
    if self.residual and self.history == True:
      assert(self.history==True)
      h_e_out = h_e_update + h_e
    else:
      h_e_out = h_e_update

    return h_e_out

if __name__ == '__main__':
  # test case

  numNode   = 10
  graphNum  = 24
  nodeDim   = 13
  edgeDim   = 15
  globalDim = 24

  edgeUpdate = EdgeFunction(nodeDim, [edgeDim, edgeDim+1], globalDim)
  edgeUpdate.cuda()

  h_v = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  h_e = Variable(torch.zeros(graphNum, numNode, numNode, edgeDim)).cuda()
  u   = Variable(torch.zeros(graphNum, globalDim)).cuda()
  V   = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  E   = Variable(torch.LongTensor(graphNum, numNode, numNode).random_(0,2).float()).cuda()

  h_e = edgeUpdate(u, h_v, h_e, V, E)

  print('h_e shape:', h_e.size())
