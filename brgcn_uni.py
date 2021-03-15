import torch
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from torch_geometric.nn.inits import glorot

class BRGCNConvNode(MessagePassing):
    def __init__(self, in_channels, out_channels, neg_slope, num_relations, heads=1, dropout=0, self_connect=True):
        super(BRGCNConvNode, self).__init__()
        ## set the layers and would be reset in self.reset_parameters function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neg_slope = neg_slope
        self.num_relations = num_relations
        self.heads = heads
        self.dropout = dropout
        self.self_connect = self_connect

        self.linear_j = Linear(in_channels, heads*out_channels, bias=False)
        self.linear_i = Linear(in_channels, heads*out_channels, bias=False)
        self.node_att = Parameter(torch.Tensor(num_relations, heads, out_channels*2))
        if self_connect:
            self.W_self_node = Parameter(torch.Tensor(in_channels, heads * out_channels))
        else:
            self.register_parameter('W_self_node', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.linear_j.reset_parameters()
        self.linear_i.reset_parameters()
        glorot(self.node_att)
        if self.self_connect:
            glorot(self.W_self_node)

    def forward(self, x, edge_index, edge_type, node_type):
        '''
        # single layer for the BR-GCN convolution, given the neighboring nodes and target nodes with their
        # embeddings and related edge type (relation type)
        :param x: (x_source, x_target)
        :param edge_index: the global edge index between source and target
        :param edge_type: all of the edge types from source to target
        :param node_type: node type of the target nodes
        :return:
        '''
        if isinstance(x,(tuple, list)):
            x_neighbor, x_target = x
        else:
            x_neighbor, x_target = x, x

        final_embeddings = x_target.new_zeros(x_target.size(0), self.heads*self.out_channels)

        h_j = self.linear_j(x_neighbor)
        h_i = self.linear_i(x_target)

        x_h = (h_j, h_i)

        for r in range(self.num_relations):
            relation_edge_mask = edge_type == r
            if relation_edge_mask.sum()==0:
                continue
            # Compute relation specific node embedding (Node Level attention)
            z_r_i = self.propagate(edge_index[:, relation_edge_mask], x=x_h, relation=r)
            if self.self_connect:
                node_mask = (z_r_i.sum(1) != 0).view(-1, 1)
                final_embeddings += z_r_i + x_target.matmul(self.W_self_node)*node_mask
            else:
                final_embeddings += z_r_i
        return final_embeddings

    def message(self, x_i, x_j, edge_index_i, relation):
        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)
        alpha = (self.node_att[relation].unsqueeze(0) * torch.cat([x_i, x_j], dim=-1)).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.neg_slope)
        alpha = softmax(alpha, edge_index_i)
        if self.training and self.dropout>0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)
        return (alpha.unsqueeze(len(alpha.size())) * x_j).view(-1, self.heads*self.out_channels)

class BRGCNConvRel(MessagePassing):
    def __init__(self, in_channels, out_channels, neg_slope, num_relations, num_node_types, heads=1, dropout=0):
        super(BRGCNConvRel, self).__init__()
        ## set the layers and would be reset in self.reset_parameters function
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neg_slope = neg_slope
        self.num_relations = num_relations
        self.num_node_types = num_node_types
        self.heads = heads
        self.dropout = dropout

        self.W = Parameter(torch.Tensor(num_relations, in_channels, heads*out_channels))

        self.W_q = Parameter(torch.Tensor(num_relations, heads*out_channels, heads*out_channels))
        self.W_k = Parameter(torch.Tensor(num_relations, heads*out_channels, heads*out_channels))
        self.W_v = Parameter(torch.Tensor(num_relations, heads*out_channels, heads*out_channels))

        self.W_self = Parameter(torch.Tensor(in_channels, heads * out_channels))

        self.W_relation = Parameter(torch.Tensor(num_relations, 1))


        self.reset_parameters()


    def reset_parameters(self):
        glorot(self.W)
        glorot(self.W_q)
        glorot(self.W_k)
        glorot(self.W_v)
        glorot(self.W_relation)
        glorot(self.W_self)

    def forward(self, x, edge_index, edge_type, node_type):
        '''
        # single layer for the BR-GCN convolution, given the neighboring nodes and target nodes with their
        # embeddings and related edge type (relation type)
        :param x: (x_source, x_target)
        :param edge_index: the global edge index between source and target
        :param edge_type: all of the edge types from source to target
        :param node_type: node type of the target nodes
        :return:
        '''
        if isinstance(x,(tuple, list)):
            x_neighbor, x_target = x
        else:
            x_neighbor, x_target = x, x

        final_embeddings = x_target.new_zeros(x_target.size(0), self.heads*self.out_channels)

        q_i = x_target.new_zeros(self.num_relations, x_target.size(0), self.heads*self.out_channels)
        k_i = x_target.new_zeros(self.num_relations, x_target.size(0), self.heads*self.out_channels)
        v_i = x_target.new_zeros(self.num_relations, x_target.size(0), self.heads*self.out_channels)

        for r in range(self.num_relations):
            relation_edge_mask = edge_type == r
            if relation_edge_mask.sum()==0:
                continue
            z_r_i = self.propagate(edge_index[:, relation_edge_mask], x=x, relation=r)

            q_i[r] = torch.matmul(z_r_i.view(-1, self.heads*self.out_channels), self.W_q[r, :, :])
            k_i[r] = torch.matmul(z_r_i.view(-1, self.heads*self.out_channels), self.W_k[r, :, :])
            v_i[r] = torch.matmul(z_r_i.view(-1, self.heads*self.out_channels), self.W_v[r, :, :])

        for r in range(self.num_relations):
            psi_r = ((q_i[r].view(-1, self.heads, self.out_channels).unsqueeze(0)) *
                     (k_i.view(self.num_relations, -1, self.heads, self.out_channels))).sum(-1)
            mask_psi = (psi_r==0)*((psi_r.sum(0)!=0).view(1, -1, self.heads))
            psi_r_prob = F.softmax(psi_r.masked_fill(mask_psi.bool(), float('-inf')), dim=0)
            delta_r = ((psi_r_prob.unsqueeze(3)) *
                       (v_i.view(self.num_relations, -1, self.heads, self.out_channels))).sum(0)
            delta_r = delta_r.view(-1, self.heads * self.out_channels)
            final_embeddings += delta_r * self.W_relation[r]
        final_embeddings += x_target.matmul(self.W_self)
        return final_embeddings

    def message(self, x_j, relation):
        out = torch.matmul(x_j, self.W[relation, :, :])
        return out