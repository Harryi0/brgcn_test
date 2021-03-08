from copy import copy

import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear, ModuleList, ParameterDict, init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from brgcn_example import BRGCN_O

from torch_geometric.nn.inits import glorot, ones

class BRGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, neg_slope, num_relations, heads=1, dropout=0):
        super(BRGCNConv, self).__init__()
        ## set the layers and would be reset in self.reset_parameters function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neg_slope = neg_slope
        self.num_relations = num_relations
        self.heads = heads
        self.dropout = dropout

        self.linear_j = Linear(in_channels, heads*out_channels, bias=False)
        self.linear_i = Linear(in_channels, heads*out_channels, bias=False)
        self.node_att = Parameter(torch.Tensor(num_relations, heads, out_channels*2))
        self.W_q = Parameter(torch.Tensor(num_relations, heads*out_channels, out_channels))
        self.W_k = Parameter(torch.Tensor(num_relations, heads*out_channels, out_channels))
        self.W_v = Parameter(torch.Tensor(num_relations, heads*out_channels, out_channels))

        self.W_self = Parameter(torch.Tensor(in_channels, out_channels))
        self.W_self_node = Parameter(torch.Tensor(in_channels, heads*out_channels))
        self.W_relation = Parameter(torch.Tensor(num_relations, 1))

        self.reset_parameters()


    def reset_parameters(self):
        self.linear_j.reset_parameters()
        self.linear_i.reset_parameters()
        glorot(self.node_att)
        glorot(self.W_q)
        glorot(self.W_k)
        glorot(self.W_v)
        glorot(self.W_self)
        glorot(self.W_self_node)
        glorot(self.W_relation)

    def forward(self, x, edge_index, edge_type):
        '''
        # single layer for the BR-GCN convolution, given the neighboring nodes and target nodes with their
        # embeddings and related edge type (relation type)
        :param x: (x_source, x_target)
        :param edge_index: the global edge index between source and target
        :param edge_type: all of the edge types from source to target
        :param node_type: node type of the target nodes
        :return:
        '''
        # TODO: note the x_neighbor also contains the target nodes
        if isinstance(x,(tuple, list)):
            x_neighbor, x_target = x
        else:
            x_neighbor, x_target = x, x

        final_embeddings = x_target.new_zeros(x_target.size(0), self.out_channels)

        # TODO: why need to pass this linear layer
        h_j = self.linear_j(x_neighbor)
        h_i = self.linear_i(x_target)

        x_h = (h_j, h_i)

        q_i = x_target.new_zeros(self.num_relations, x_target.size(0), self.out_channels)
        k_i = x_target.new_zeros(self.num_relations, x_target.size(0), self.out_channels)
        v_i = x_target.new_zeros(self.num_relations, x_target.size(0), self.out_channels)

        for r in range(self.num_relations):
        ### relation-specific node neighbors
        ### node level attention for each relation specific neighbors
        ### relation specific node embedding
            relation_edge_mask = edge_type == r
            if relation_edge_mask.sum()==0:
                continue
            # Compute relation specific node embedding (Node Level attention)
            # [num_nodes, heads, out_channels]
            z_r_i = self.propagate(edge_index[:, relation_edge_mask], x=x_h, relation=r) \
                    + x_target.matmul(self.W_self_node)
        ### construct query key value matrices
            q_i[r] = torch.matmul(z_r_i.view(-1, self.heads*self.out_channels), self.W_q[r, :, :])
            k_i[r] = torch.matmul(z_r_i.view(-1, self.heads*self.out_channels), self.W_k[r, :, :])
            v_i[r] = torch.matmul(z_r_i.view(-1, self.heads*self.out_channels), self.W_v[r, :, :])

        # output = self.aggregate_relation_attention(q_i, k_i, v_i, h_i, edge_index)
        for r in range(self.num_relations):
            psi_r = (q_i[r,:,:].unsqueeze(0) * k_i).sum(-1)
            psi_r = F.softmax(psi_r, dim=0)
            delta_r = (psi_r.unsqueeze(2) * v_i).sum(0)
            mask = (delta_r.sum(1) != 0).view(-1, 1)
            embed_r = delta_r + x_target.matmul(self.W_self)*mask
            final_embeddings += embed_r * self.W_relation[r]
            # delta_r = (psi_r.unsqueeze(2) * v_i + x_i.matmul(self.W_self)).sum(0)
            # mask = (delta_r.sum(1)!=0).view(-1,1)

            # embed_r = F.softmax(delta_r + torch.matmul(x_target, self.W_self), dim=1)*mask


        return final_embeddings

    def message(self, x_i, x_j, edge_index_i, relation):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (self.node_att[relation].unsqueeze(0) * torch.cat([x_i, x_j], dim=-1)).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.neg_slope)
        alpha = softmax(alpha, edge_index_i)
        # TODO: Possible drop out step at here
        if self.training and self.dropout>0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        return (alpha.unsqueeze(len(alpha.size())) * x_j).view(-1, self.heads*self.out_channels)

class BRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, neg_slope, heads,
                 num_relations, num_nodes_dict, x_types):
        super(BRGCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        self.num_relations = num_relations

        node_types = list(num_nodes_dict.keys())



        self.emb_dict = ParameterDict({
            f'{key}': Parameter(torch.Tensor(num_nodes_dict[key], in_channels))
            for key in set(node_types).difference(set(x_types))
        })

        self.convs = ModuleList()
        self.convs.append(BRGCNConv(self.in_channels, self.hidden_channels, neg_slope,
                                    self.num_relations, self.heads, self.dropout))
        for _ in  range(num_layers-2):
            self.convs.append(BRGCNConv(self.hidden_channels, self.hidden_channels, neg_slope, self.num_relations,
                                        self.heads, self.dropout))
        self.convs.append(BRGCNConv(self.hidden_channels, self.out_channels, neg_slope,
                                    self.num_relations, self.heads, self.dropout))

        self.reset_parameters()
        # (self, in_channels, out_channels, neg_slope, num_relations, heads=1, dropout=0):

    def reset_parameters(self):
        #TODO: look into the embeddinng intialization
        for embedding in self.emb_dict.values():
            init.xavier_uniform_(embedding)
        for conv in self.convs:
            conv.reset_parameters()

    def group_input(self, x_dict, node_type, local_node_idx, n_id=None):
        # Create global node feature matrix.
        if n_id is not None:
            node_type = node_type[n_id]
            local_node_idx = local_node_idx[n_id]

        h = torch.zeros((node_type.size(0), self.in_channels),
                        device=node_type.device)

        for key, x in x_dict.items():
            mask = node_type == key
            h[mask] = x[local_node_idx[mask]]

        for key, emb in self.emb_dict.items():
            mask = node_type == int(key)
            h[mask] = emb[local_node_idx[mask]]

        return h

    def forward(self, n_id, x_dict, adjs, edge_type, node_type,
                local_node_idx):
        '''
        :param n_id: node index for the source nodes
        :param x_dict: Node embedding dictionary with node type  as key
        :param adjs: source to node structure, (edge_index, e_id, size)
        :param edge_type: the edge type for all edges
        :param node_type: the node type for all nodes
        :param local_node_idx: transform the global node id to a local one order for each type of node
        :return:
        '''
        x = self.group_input(x_dict, node_type, local_node_idx, n_id)
        node_type = node_type[n_id]

        for i, (edge_index, e_id, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target node embeddings.
            node_type = node_type[:size[1]]  # Target node types.
            conv = self.convs[i]
            x = conv((x, x_target), edge_index, edge_type[e_id])
            if i != self.num_layers - 1:
                x = F.relu(x)
                # x = F.dropout(x, p=self.dropout, training=self.training)

        return x.log_softmax(dim=-1)

    def fullBatch_inference(self, x_dict, edge_index, edge_type, node_type, local_node_idx):
        x = self.group_input(x_dict, node_type, local_node_idx)
        node_type = node_type[edge_index[1].unique()]
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type, node_type)
            if i!=self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def group_inference(self, x_dict, edge_index_dict, key2int):
        x_dict = copy(x_dict)
        for k, emb in self.emb_dict.items():
            x_dict[int(k)] = emb
        for i, conv in enumerate(self.convs):
            out_dict = dict()
            for n, x_target in x_dict.items():
                edge_index_n = []
                edge_type_n = []
                node_type_nbr = []
                x_nbr = []
                for k, e_i in edge_index_dict.items():
                    if key2int[k[-1]] == n:
                        edge_index_n.append(e_i)
                        edge_type_n.append(e_i.new_full((e_i.size(1),), key2int[k]))
                        node_type_nbr.append(e_i.new_full((x_dict[key2int[k[0]]].size(0),), key2int[k[0]]))
                        x_nbr.append(x_dict[key2int[k[0]]])
                edge_index_n = torch.cat(edge_index_n, dim=1)
                edge_type_n = torch.cat(edge_type_n, dim=0)
                x = torch.cat([x_target]+x_nbr, dim=0)
                node_type_target = edge_index_n.new_full((x_target.size(0),), n)
                node_type_n = torch.cat([node_type_target]+node_type_nbr, dim=0)
                x_out = conv((x, x_target), edge_index_n, edge_type_n, node_type_n)
                if i != self.num_layers - 1:
                    x_out = F.relu(x_out)
                    x_out = F.dropout(x_out, p=self.dropout, training=self.training)
                out_dict[n] = x_out
            # if i != self.num_layers - 1:
            #     for j in range(self.num_node_types):
            #         F.relu_(out_dict[j])
            x_dict = out_dict
        return x_dict