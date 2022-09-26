import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class GlobalFunction(nn.Module):
  '''
    Input:  (U, {h_v}, {h_e}), (V,E)
    Output: (U, {h_v}, {h_e})
  '''
  def __init__(self, nodedim, edgedim, globaldims, residual=False):
    super(GlobalFunction, self).__init__()

    self.nodedim_in   = nodedim
    self.edgedim_in   = edgedim
    self.globaldim_in  = globaldims[0]
    self.globaldim_out = globaldims[1]
    self.residual      = residual

    ''' message updates function for edges '''

    self.globalInfoDim = self.nodedim_in + self.edgedim_in + self.globaldim_in
    self.edgeAggregator = nn.Sequential(
                            nn.Conv2d(self.edgedim_in, self.edgedim_in, 1, 1),
                            nn.ELU(),
                            nn.Conv2d(self.edgedim_in, self.edgedim_in, 1, 1)
                          )
    self.nodeAggregator = nn.Sequential(
                            nn.Conv1d(self.nodedim_in, self.nodedim_in, 1, 1),
                            nn.ELU(),
                            nn.Conv1d(self.nodedim_in, self.nodedim_in, 1, 1)
                          )
    self.function  = nn.Sequential(
                       nn.Linear(self.globalInfoDim, self.globalInfoDim),
                       nn.ELU(),
                       nn.Linear(self.globalInfoDim, self.globaldim_out)
                     )

  def forward(self, u, h_v, h_e, V, E):
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

    edgeGlobalFeature = self.edgeAggregator(h_e.transpose(2,3).transpose(1,2))
    E_mask = E[:,:,:,0:1].expand(h_e.shape).transpose(2,3).transpose(1,2)
    edgeGlobalFeature = (E_mask * edgeGlobalFeature).sum(2).sum(2)

    nodeGlobalFeature = self.nodeAggregator(h_v.transpose(1,2))
    V_mask = 1
    if V is not None:
        V_mask = V.unsqueeze(-1).expand(h_v.shape).transpose(1,2)
    nodeGlobalFeature = (V_mask * nodeGlobalFeature).sum(2)
 
    globalFeature = torch.cat([u, edgeGlobalFeature, nodeGlobalFeature], dim=1)
    u_update = self.function(globalFeature)

    if self.residual:
      u = u_update + u
    else:
      u = u_update

    return u

if __name__ == '__main__':
  # test case

  numNode   = 10
  graphNum  = 24
  nodeDim   = 13
  edgeDim   = 15
  globalDim = 26

  globalUpdate = GlobalFunction(nodeDim, edgeDim, [globalDim, globalDim+1])
  globalUpdate.cuda()

  h_v = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  h_e = Variable(torch.zeros(graphNum, numNode, numNode, edgeDim)).cuda()
  u   = Variable(torch.zeros(graphNum, globalDim)).cuda()
  V   = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  E   = Variable(torch.LongTensor(graphNum, numNode, numNode).random_(0,2).float()).cuda()

  u = globalUpdate(u, h_v, h_e, V, E)

  print('u size:', u.size())
