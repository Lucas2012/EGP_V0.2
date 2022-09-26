import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pdb

class NodeFunction(nn.Module):
  '''
    Input:  (U, {h_v}, {h_e}), (V,E)
    Output: (U, {h_v}, {h_e})
  '''
  def __init__(self, nodedims, edgedim, globaldim, residual=False, history=True, use_global=True, rnn=False, message_normalize=False, dropout=0):
    super(NodeFunction, self).__init__()

    self.nodedim_in   = nodedims[0]
    self.nodedim_out  = nodedims[1]
    self.edgedim_in   = edgedim
    self.globaldim_in = globaldim
    self.residual     = residual
    self.history      = history
    self.use_global   = use_global
    self.message_normalize = message_normalize

    ''' message updates function for nodes '''

    self.nodeInfoDim    = self.edgedim_in * 2
    if self.history and rnn == False:
      self.nodeInfoDim    += self.nodedim_in
    if self.use_global:
      self.nodeInfoDim    += self.globaldim_in
    self.edgeSrcMapper = nn.Sequential(
                            nn.Linear(self.edgedim_in, self.edgedim_in),
                            nn.ELU(),
                            #nn.Dropout(p=dropout),
                            nn.Linear(self.edgedim_in, self.edgedim_in)
                         )
    self.edgeTgtMapper = nn.Sequential(
                            nn.Linear(self.edgedim_in, self.edgedim_in),
                            nn.ELU(),
                            #nn.Dropout(p=dropout),
                            nn.Linear(self.edgedim_in, self.edgedim_in)
                         )

    self.rnn = rnn
    if rnn == False:
      self.function  = nn.Conv1d(self.nodeInfoDim, self.nodedim_out, 1, 1)
    else:
      assert 1 == 0
      self.function  = nn.GRUCell(self.nodeInfoDim, self.nodedim_out)

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
         node   state -> h_v: [K, |V|, nodedim]
    '''

    graphNum, numNode = h_v.size(0), h_v.size(1)
    #h_e_conv = h_e.transpose(2,3).transpose(1,2)
    
    # aggregate edge information per node
    srcEdgeInfo = torch.zeros(graphNum, numNode, h_e.shape[-1]).cuda()
    tgtEdgeInfo = torch.zeros(graphNum, numNode, h_e.shape[-1]).cuda()
    if self.message_normalize:
        E = E[:,:,:,1:].sum(-1)
        E = (E > 0).float()
        srcEdgeInfo = h_e_conv.sum(2)
        srcEdgeInfo = srcEdgeInfo / (E.sum(1).unsqueeze(1).expand(srcEdgeInfo.size()) + 1)
        srcEdgeInfo = self.edgeSrcMapper(srcEdgeInfo)
    
        tgtEdgeInfo = h_e_conv.sum(3)
        tgtEdgeInfo = tgtEdgeInfo / (E.sum(2).unsqueeze(1).expand(tgtEdgeInfo.size()) + 1)
        tgtEdgeInfo = self.edgeTgtMapper(tgtEdgeInfo)
    else:
        E_src_sum   = E[:,:,:,1:].sum(-1).sum(1)
        srcEdgeInfo[E_src_sum > 0] = self.edgeSrcMapper(h_e.sum(1)[E_src_sum > 0]) # / E.sum(1)
        srcEdgeInfo = srcEdgeInfo.transpose(1,2)

        E_tgt_sum   = E[:,:,:,1:].sum(-1).sum(2)
        tgtEdgeInfo[E_tgt_sum > 0] = self.edgeTgtMapper(h_e.sum(2)[E_tgt_sum > 0]) # / E.sum(2)
        tgtEdgeInfo = tgtEdgeInfo.transpose(1,2)

    info_vec = [srcEdgeInfo, tgtEdgeInfo]
    if self.use_global:
      globalGrid  = u.unsqueeze(2).expand(u.size(0), u.size(1), numNode)
      info_vec += [globalGrid]
    if self.history and self.rnn == False:
      info_vec += [h_v.transpose(1,2)]

    nodeFeature = torch.cat(info_vec, dim=1)

    if not self.rnn:
      # node update function
      h_v_update = self.function(nodeFeature)
      h_v_update = h_v_update.transpose(1,2)
    else:
      nodeFeature = nodeFeature.transpose(1,2)
      h_v_update = self.function(nodeFeature.contiguous().view(-1, nodeFeature.size(-1)), h_v.view(-1, h_v.size(-1)))
      h_v_update = h_v_update.view(-1, numNode, h_v_update.size(-1))

    if self.residual and self.rnn == False and self.history == True:
      h_v = h_v_update + h_v
    else:
      h_v = h_v_update

    if V is not None:
      h_v = h_v * V.unsqueeze(-1).expand(h_v.shape)

    return h_v

if __name__ == '__main__':
  # test case

  numNode   = 10
  graphNum  = 24
  nodeDim   = 13
  edgeDim   = 15
  globalDim = 24

  nodeUpdate = NodeFunction([nodeDim, nodeDim+1], edgeDim, globalDim)
  nodeUpdate.cuda()

  h_v = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  h_e = Variable(torch.zeros(graphNum, numNode, numNode, edgeDim)).cuda()
  u   = Variable(torch.zeros(graphNum, globalDim)).cuda()
  V   = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  E   = Variable(torch.LongTensor(graphNum, numNode, numNode).random_(0,2).float()).cuda()

  h_v = nodeUpdate(u, h_v, h_e, V, E)

  print('h_v shape:', h_v.size())
