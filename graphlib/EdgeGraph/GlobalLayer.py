import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import weight_norm
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class GlobalLayer(nn.Module):
  '''
    Layer for global aggregation.

    Input:  (U, {h_v}, {h_e}), (V,E)
    Output: (U, {h_v}, {h_e})
  '''
  def __init__(
          self,
          nodedim,
          edgedim,
          globaldim,
          outdim
    ):
    super(GlobalLayer, self).__init__()

    self.nodedim   = nodedim
    self.edgedim   = edgedim
    self.globaldim = globaldim
    self.outdim    = outdim

    ''' message updates functions '''
    # edge updates
    self.edgeGate = nn.Sequential(
                             nn.Conv2d(edgedim, outdim*2, 1, 1),
                           ) 

    # node updates
    self.nodeGate = nn.Sequential(
                             nn.Conv1d(nodedim, outdim*2, 1, 1),
                           )

    # global updates
    self.globalGate = nn.Sequential(
                               nn.Linear(globaldim, outdim*2),
                             ) 

  def forward(self, graph_states, graph):
    '''
      * K graphs per forward, all graphs share the same name space for node and edge

      input: 
         global state -> u:   [1, globaldim]
         node   state -> h_v: [1, |V|, nodedim]
         edge   state -> h_e: [1, |V|, |V|, edgedim]

         V -> nodes mask: [1, |V|]
         E -> edges mask: [1, |V|, |V|]

      output: 
         global state -> u:   [1, globaldim]
         node   state -> h_v: [1, |V|, nodedim]
         edge   state -> h_e: [1, |V|, |V|, edgedim]
    '''

    # inputs
    [u, h_v, h_e] = graph_states
    [V, E]        = graph

    # edge updates
    h_e  = h_e.transpose(2,3).transpose(1,2)
    edge     = self.edgeGate(h_e)
    edgeGate = F.sigmoid(edge[:,:self.outdim,:,:])
    edgeInfo = edge[:,self.outdim:,:,:]

    # node updates
    h_v = h_v.transpose(1,2)
    node   = self.nodeGate(h_v)
    nodeGate = F.sigmoid(node[:,:self.outdim,:])
    nodeInfo = node[:,self.outdim:,:]

    # global updates
    glo  = self.globalGate(u)
    globalGate = F.sigmoid(glo[:,:self.outdim])
    globalInfo = glo[:,self.outdim:]

    output = globalGate * globalInfo + (nodeGate * nodeInfo).sum(2) + (edgeGate * edgeInfo).sum(2).sum(2)

    return output


if __name__ == '__main__':
  # test case

  graphNum  = 1
  numNode   = 10
  nodeDim   = 13
  edgeDim   = 15
  globalDim = 24
  queryDim  = 19
  outDim    = 32

  '''
  graphGlobal = GlobalLayer(nodeDim, edgeDim, globalDim, querydim=queryDim, outdim=outDim)
  graphGlobal.cuda()

  h_v = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  h_e = Variable(torch.zeros(graphNum, numNode, numNode, edgeDim)).cuda()
  u   = Variable(torch.zeros(graphNum, globalDim)).cuda()
  V   = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  E   = Variable(torch.LongTensor(graphNum, numNode, numNode).random_(0,2).float()).cuda()

  query   = Variable(torch.zeros(graphNum, queryDim)).cuda()

  output = graphGlobal([u, h_v, h_e], [V, E], query)
  '''

  graphGlobal = GlobalLayer(nodeDim, edgeDim, globalDim, outdim=outDim)
  graphGlobal.cuda()

  h_v = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  h_e = Variable(torch.zeros(graphNum, numNode, numNode, edgeDim)).cuda()
  u   = Variable(torch.zeros(graphNum, globalDim)).cuda()
  V   = Variable(torch.zeros(graphNum, numNode, nodeDim)).cuda()
  E   = Variable(torch.LongTensor(graphNum, numNode, numNode).random_(0,2).float()).cuda()

  output = graphGlobal([u, h_v, h_e], [V, E])
  print('output dimensions:', output.size())
