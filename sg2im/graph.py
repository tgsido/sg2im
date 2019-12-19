#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from sg2im.layers import build_mlp

"""
PyTorch modules for dealing with graphs.
"""

def _init_weights(module):
  if hasattr(module, 'weight'):
    if isinstance(module, nn.Linear):
      nn.init.kaiming_normal_(module.weight)

class GraphAttnConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphAttnConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

    self.initial_obj_projection_layer = nn.Linear(self.output_dim, self.hidden_dim)
    nn.init.kaiming_normal_(self.initial_obj_projection_layer.weight)

    self.W_sim = nn.Linear(self.hidden_dim, self.hidden_dim)
    nn.init.kaiming_normal_(self.W_sim.weight)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)

    prev_obj_vecs = self.initial_obj_projection_layer(obj_vecs) # (O, H)

    for i in range(O):
        s_mask = i == s_idx_exp
        o_mask = i == o_idx_exp
        s_indices_for_ith_object = None
        o_indices_for_ith_object = None
        all_vecs_for_ith_object = torch.zeros((1,H),device=device)
        if torch.nonzero(s_mask).size()[0] > 0:
          s_indices_for_ith_object  = torch.nonzero(i == s_idx).reshape(-1)
          s_vecs_for_ith_object = torch.index_select(new_s_vecs, 0, s_indices_for_ith_object) # (N,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, s_vecs_for_ith_object], dim=0)
        if torch.nonzero(o_mask).size()[0] > 0:
          o_indices_for_ith_object  = torch.nonzero(i == o_idx).reshape(-1)
          o_vecs_for_ith_object = torch.index_select(new_o_vecs, 0, o_indices_for_ith_object) # (M,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, o_vecs_for_ith_object], dim=0)

        all_vecs_proj = self.W_sim(all_vecs_for_ith_object) # (N+M, H)
        prev_obj_vec_proj = self.W_sim(prev_obj_vecs[i]) # (H,)
        sim_vector = torch.mm(all_vecs_proj, prev_obj_vec_proj.unsqueeze(1)) #(N+M, 1)
        softmax_layer = nn.Softmax(dim=0)
        sim_vector = softmax_layer(sim_vector)
        sim_mask = sim_vector #(N+M,1)
        scaled_all_vecs = all_vecs_for_ith_object * sim_mask # (N+M, H)
        pooled_obj_vecs[i,:] = torch.sum(scaled_all_vecs, dim=0) # (H,)


    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs

class GraphSageMeanConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphSageMeanConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

    self.initial_obj_projection_layer = nn.Linear(self.output_dim, self.hidden_dim)
    nn.init.kaiming_normal_(self.initial_obj_projection_layer.weight)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)

    prev_obj_vecs = self.initial_obj_projection_layer(obj_vecs) # (O, H)
    for i in range(O):
        s_mask = i == s_idx_exp
        o_mask = i == o_idx_exp
        s_indices_for_ith_object = None
        o_indices_for_ith_object = None
        all_vecs_for_ith_object = torch.zeros((1,H),device=device)
        if torch.nonzero(s_mask).size()[0] > 0:
          s_indices_for_ith_object  = torch.nonzero(i == s_idx).reshape(-1)
          s_vecs_for_ith_object = torch.index_select(new_s_vecs, 0, s_indices_for_ith_object) # (N,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, s_vecs_for_ith_object], dim=0)
        if torch.nonzero(o_mask).size()[0] > 0:
          o_indices_for_ith_object  = torch.nonzero(i == o_idx).reshape(-1)
          o_vecs_for_ith_object = torch.index_select(new_o_vecs, 0, o_indices_for_ith_object) # (M,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, o_vecs_for_ith_object], dim=0)
        old_vec_and_neighborhood_stacked = torch.cat([all_vecs_for_ith_object, prev_obj_vecs[i].unsqueeze(0)], dim=0)
        pooled_obj_vecs[i,:] = torch.mean(old_vec_and_neighborhood_stacked, dim=0) # (H,)

    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphSageLSTMConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphSageLSTMConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    self.output_linear_layer = nn.Linear(self.hidden_dim * 2, self.output_dim)
    nn.init.kaiming_normal_(self.output_linear_layer.weight)

    self.initial_obj_projection_layer = nn.Linear(self.output_dim, self.hidden_dim)
    nn.init.kaiming_normal_(self.initial_obj_projection_layer.weight)

    ### RNN Component ###
    self.object_lstm = nn.LSTM(
                        input_size=self.hidden_dim,
                        hidden_size=self.hidden_dim,
                        num_layers=1
                        )

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)

    for i in range(O):
        s_mask = i == s_idx_exp
        o_mask = i == o_idx_exp
        s_indices_for_ith_object = None
        o_indices_for_ith_object = None
        all_vecs_for_ith_object = torch.zeros((1,H),device=device)
        if torch.nonzero(s_mask).size()[0] > 0:
          s_indices_for_ith_object  = torch.nonzero(i == s_idx).reshape(-1)
          s_vecs_for_ith_object = torch.index_select(new_s_vecs, 0, s_indices_for_ith_object) # (N,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, s_vecs_for_ith_object], dim=0)
        if torch.nonzero(o_mask).size()[0] > 0:
          o_indices_for_ith_object  = torch.nonzero(i == o_idx).reshape(-1)
          o_vecs_for_ith_object = torch.index_select(new_o_vecs, 0, o_indices_for_ith_object) # (M,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, o_vecs_for_ith_object], dim=0)

        ## shuffle neighboring vecs ##
        all_vecs_for_ith_object=all_vecs_for_ith_object[torch.randperm(all_vecs_for_ith_object.size()[0])]
        lstm_input = all_vecs_for_ith_object.unsqueeze(1)
        output, (h_n, c_n) = self.object_lstm(lstm_input)
        pooled_obj_vecs[i,:] = h_n.reshape(-1)

    prev_obj_vecs = self.initial_obj_projection_layer(obj_vecs) # (O, H)
    new_old_obj_vecs_stack = torch.cat([pooled_obj_vecs, prev_obj_vecs], dim=1) #  shape: (O,  2H)
    new_obj_vecs = self.output_linear_layer(new_old_obj_vecs_stack) # (O, Dout)

    return new_obj_vecs, new_p_vecs

class GraphSageMaxPoolConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphSageMaxPoolConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    self.output_linear_layer = nn.Linear(self.hidden_dim * 2, self.output_dim)
    nn.init.kaiming_normal_(self.output_linear_layer.weight)

    self.initial_obj_projection_layer = nn.Linear(self.output_dim, self.hidden_dim)
    nn.init.kaiming_normal_(self.initial_obj_projection_layer.weight)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)

    for i in range(O):
        s_mask = i == s_idx_exp
        o_mask = i == o_idx_exp
        s_indices_for_ith_object = None
        o_indices_for_ith_object = None
        all_vecs_for_ith_object = torch.zeros((1,H),device=device)
        if torch.nonzero(s_mask).size()[0] > 0:
          s_indices_for_ith_object  = torch.nonzero(i == s_idx).reshape(-1)
          s_vecs_for_ith_object = torch.index_select(new_s_vecs, 0, s_indices_for_ith_object) # (N,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, s_vecs_for_ith_object], dim=0)
        if torch.nonzero(o_mask).size()[0] > 0:
          o_indices_for_ith_object  = torch.nonzero(i == o_idx).reshape(-1)
          o_vecs_for_ith_object = torch.index_select(new_o_vecs, 0, o_indices_for_ith_object) # (M,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, o_vecs_for_ith_object], dim=0)
        max_v, max_indices = torch.max(all_vecs_for_ith_object, dim=0)
        pooled_obj_vecs[i,:] = max_v

    prev_obj_vecs = self.initial_obj_projection_layer(obj_vecs) # (O, H)
    new_old_obj_vecs_stack = torch.cat([pooled_obj_vecs, prev_obj_vecs], dim=1) #  shape: (O,  2H)
    new_obj_vecs = self.output_linear_layer(new_old_obj_vecs_stack) # (O, Dout)

    return new_obj_vecs, new_p_vecs

class GraphTripleRandomWalkConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphTripleRandomWalkConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    ### Add Random Walk by creating random edges to incorporate more global information ###
    num_random_edges = max(1, O // 10)

    rand_edges = torch.randint(O, (num_random_edges, 2), dtype=torch.long,device=device)
    s_idx_rand = rand_edges[:,0].contiguous()
    o_idx_rand = rand_edges[:,1].contiguous()

    s_idx_rand_walk = torch.cat([s_idx,s_idx_rand],dim=0)
    o_idx_rand_walk = torch.cat([o_idx,o_idx_rand],dim=0)
    permutation = torch.randperm(T + num_random_edges)
    new_index = permutation[:T]
    s_idx = s_idx_rand_walk[new_index]
    o_idx = o_idx_rand_walk[new_index]

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(O, dtype=dtype, device=device)
      ones = torch.ones(T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)

      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (O, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs

class GraphTripleRnnConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphTripleRnnConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

    ### RNN Component ###
    self.object_rnn = nn.RNN(self.hidden_dim, self.hidden_dim)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)

    for i in range(O):
        s_mask = i == s_idx_exp
        o_mask = i == o_idx_exp
        s_indices_for_ith_object = None
        o_indices_for_ith_object = None
        all_vecs_for_ith_object = torch.zeros((1,H),device=device)
        if torch.nonzero(s_mask).size()[0] > 0:
          s_indices_for_ith_object  = torch.nonzero(i == s_idx).reshape(-1)
          s_vecs_for_ith_object = torch.index_select(new_s_vecs, 0, s_indices_for_ith_object) # (N,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, s_vecs_for_ith_object], dim=0)
        if torch.nonzero(o_mask).size()[0] > 0:
          o_indices_for_ith_object  = torch.nonzero(i == o_idx).reshape(-1)
          o_vecs_for_ith_object = torch.index_select(new_o_vecs, 0, o_indices_for_ith_object) # (M,H)
          all_vecs_for_ith_object = torch.cat([all_vecs_for_ith_object, o_vecs_for_ith_object], dim=0)

        ## run through rnn ##
        rnn_input = all_vecs_for_ith_object.unsqueeze(1)
        output, h_n = self.object_rnn(rnn_input)
        pooled_obj_vecs[i,:] = h_n.reshape(-1)

    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs

class GraphTripleConv(nn.Module):
  """
  A single layer of scene graph convolution.
  """
  def __init__(self, input_dim, output_dim=None, hidden_dim=512,
               pooling='avg', mlp_normalization='none',model_type=None):
    super(GraphTripleConv, self).__init__()
    if output_dim is None:
      output_dim = input_dim
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.hidden_dim = hidden_dim

    assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
    self.pooling = pooling
    net1_layers = [3 * input_dim, hidden_dim, 2 * hidden_dim + output_dim]
    net1_layers = [l for l in net1_layers if l is not None]
    self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
    self.net1.apply(_init_weights)

    net2_layers = [hidden_dim, hidden_dim, output_dim]
    self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
    self.net2.apply(_init_weights)

  def forward(self, obj_vecs, pred_vecs, edges):
    """
    Inputs:
    - obj_vecs: FloatTensor of shape (O, D) giving vectors for all objects
    - pred_vecs: FloatTensor of shape (T, D) giving vectors for all predicates
    - edges: LongTensor of shape (T, 2) where edges[k] = [i, j] indicates the
      presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

    Outputs:
    - new_obj_vecs: FloatTensor of shape (O, D) giving new vectors for objects
    - new_pred_vecs: FloatTensor of shape (T, D) giving new vectors for predicates
    """
    dtype, device = obj_vecs.dtype, obj_vecs.device
    O, T = obj_vecs.size(0), pred_vecs.size(0)
    Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim

    # Break apart indices for subjects and objects; these have shape (T,)
    s_idx = edges[:, 0].contiguous()
    o_idx = edges[:, 1].contiguous()

    # Get current vectors for subjects and objects; these have shape (T, Din)
    cur_s_vecs = obj_vecs[s_idx]
    cur_o_vecs = obj_vecs[o_idx]

    # Get current vectors for triples; shape is (T, 3 * Din)
    # Pass through net1 to get new triple vecs; shape is (T, 2 * H + Dout)
    cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
    new_t_vecs = self.net1(cur_t_vecs)

    # Break apart into new s, p, and o vecs; s and o vecs have shape (T, H) and
    # p vecs have shape (T, Dout)
    new_s_vecs = new_t_vecs[:, :H]
    new_p_vecs = new_t_vecs[:, H:(H+Dout)]
    new_o_vecs = new_t_vecs[:, (H+Dout):(2 * H + Dout)]

    # Allocate space for pooled object vectors of shape (O, H)
    pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)

    # Use scatter_add to sum vectors for objects that appear in multiple triples;
    # we first need to expand the indices to have shape (T, D)
    s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
    o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
    pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

    if self.pooling == 'avg':
      # Figure out how many times each object has appeared, again using
      # some scatter_add trickery.
      obj_counts = torch.zeros(O, dtype=dtype, device=device)
      ones = torch.ones(T, dtype=dtype, device=device)
      obj_counts = obj_counts.scatter_add(0, s_idx, ones)
      obj_counts = obj_counts.scatter_add(0, o_idx, ones)

      # Divide the new object vectors by the number of times they
      # appeared, but first clamp at 1 to avoid dividing by zero;
      # objects that appear in no triples will have output vector 0
      # so this will not affect them.
      obj_counts = obj_counts.clamp(min=1)
      pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

    # Send pooled object vectors through net2 to get output object vectors,
    # of shape (O, Dout)
    new_obj_vecs = self.net2(pooled_obj_vecs)

    return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
  """ A sequence of scene graph convolution layers  """
  def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
               mlp_normalization='none', model_type=None):
    super(GraphTripleConvNet, self).__init__()

    self.num_layers = num_layers
    self.gconvs = nn.ModuleList()
    gconv_kwargs = {
      'input_dim': input_dim,
      'hidden_dim': hidden_dim,
      'pooling': pooling,
      'mlp_normalization': mlp_normalization,
      'model_type': model_type,
    }

    model_constructor = None
    if model_type == 'baseline':
        model_constructor = GraphTripleConv
    elif model_type == 'random-walk-baseline':
        model_constructor = GraphTripleRandomWalkConv
    elif model_type == 'rnn-baseline':
        model_constructor = GraphTripleRnnConv
    elif model_type == 'graphsage-maxpool':
        model_constructor = GraphSageMaxPoolConv
    elif model_type == 'graphsage-lstm':
        model_constructor = GraphSageLSTMConv
    elif model_type == 'graphsage-mean':
        model_constructor = GraphSageMeanConv
    elif model_type == 'gat-baseline':
        model_constructor = GraphAttnConv

    for _ in range(self.num_layers):
      self.gconvs.append(model_constructor(**gconv_kwargs))

  def forward(self, obj_vecs, pred_vecs, edges):
    for i in range(self.num_layers):
      gconv = self.gconvs[i]
      obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
    return obj_vecs, pred_vecs
