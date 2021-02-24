import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import to_undirected
from torch_geometric.utils.hetero import group_hetero_graph
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from tqdm import tqdm
from torch_geometric.datasets import Entities

from brgcn import BRGCN

from logger import Logger

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
# logger = Logger(args.runs, args)

print(data)

edge_index_dict = data.edge_index_dict

# We need to add reverse edges to the heterogeneous graph.
r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
edge_index_dict[('paper', 'cites', 'paper')] = edge_index

# We convert the individual graphs into a single big one, so that sampling
# neighbors does not need to care about different edge types.
# This will return the following:
# * `edge_index`: The new global edge connectivity.
# * `edge_type`: The edge type for each edge.
# * `node_type`: The node type for each node.
# * `local_node_idx`: The original index for each node.
# * `local2global`: A dictionary mapping original (local) node indices of
#    type `key` to global ones.
# `key2int`: A dictionary that maps original keys to their new canonical type.
out = group_hetero_graph(data.edge_index_dict, data.num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

# Map informations to their canonical type.
x_dict = {}
for key, x in data.x_dict.items():
    x_dict[key2int[key]] = x

num_nodes_dict = {}
for key, N in data.num_nodes_dict.items():
    num_nodes_dict[key2int[key]] = N

# Next, we create a train sampler that only iterates over the respective
# paper training nodes.
paper_idx = local2global['paper']
paper_train_idx = paper_idx[split_idx['train']['paper']]

train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[20, 10], batch_size=1024, shuffle=True) #num_workers=12)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# in_channels, hidden_channels, out_channels, num_layers, dropout, neg_slope, heads,
#                  num_relations):
brgcn = BRGCN(in_channels=128, hidden_channels=16, out_channels=dataset.num_classes, num_layers=2, dropout=0.2,
              neg_slope=0.2, heads=1, num_relations=len(edge_index_dict.keys()), num_nodes_dict=num_nodes_dict,
              x_types=list(x_dict.keys())).to(device)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# brgcn.reset_parameteres()
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)

epochs = 10
optimizer = torch.optim.Adam(brgcn.parameters(), lr=0.01)
# brgcn.reset_parameteres()
for epoch in range(1, epochs+1):
    pbar = tqdm(total=paper_train_idx.size(0))
    pbar.set_description(f'Epoch {epoch}')

    brgcn.train()
    total_loss = 0
    for batch_size, n_id, adjs, in train_loader:
        optimizer.zero_grad()

        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]

        out = brgcn(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y = y_global[n_id][:batch_size].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss * batch_size
        pbar.update(batch_size)
    pbar.close()
    epoch_loss = total_loss/paper_train_idx.size(0)
    print(f"Training epoch {epoch}/{epochs},  loss: {epoch_loss: .4f}")


