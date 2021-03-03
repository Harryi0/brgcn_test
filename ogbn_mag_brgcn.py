import argparse

import torch
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.utils import to_undirected
from torch_geometric.utils.hetero import group_hetero_graph
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from tqdm import tqdm
from torch_geometric.datasets import Entities
from util import process_mag

from brgcn import BRGCN

from logger import Logger
parser = argparse.ArgumentParser(description='BRGCN on OGBN-MAG')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--neg_slope', type=float, default=0.2)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--runs', type=int, default=1)
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset(name='ogbn-mag')
data = dataset[0]
idx_split = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-mag')
# logger = Logger(args.runs, args)

print(data)

edge_index_dict = data.edge_index_dict

# edge_index, edge_type, node_type, global2local, local2global, node_relation2int = process_mag(data)

r, c = edge_index_dict[('author', 'affiliated_with', 'institution')]
edge_index_dict[('institution', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('author', 'writes', 'paper')]
edge_index_dict[('paper', 'to', 'author')] = torch.stack([c, r])

r, c = edge_index_dict[('paper', 'has_topic', 'field_of_study')]
edge_index_dict[('field_of_study', 'to', 'paper')] = torch.stack([c, r])

# Convert to undirected paper <-> paper relation.
edge_index = to_undirected(edge_index_dict[('paper', 'cites', 'paper')])
edge_index_dict[('paper', 'cites', 'paper')] = edge_index

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
paper_train_idx = paper_idx[idx_split['train']['paper']]
paper_valid_idx = paper_idx[idx_split['valid']['paper']]
paper_test_idx = paper_idx[idx_split['test']['paper']]

train_loader = NeighborSampler(edge_index, node_idx=paper_train_idx,
                               sizes=[20, 10], batch_size=1024, shuffle=True) #num_workers=12)

valid_loader = NeighborSampler(edge_index, node_idx=paper_valid_idx,
                               sizes=[20, 10], batch_size=len(paper_valid_idx), shuffle=True) #num_workers=12)

test_loader = NeighborSampler(edge_index, node_idx=paper_test_idx,
                               sizes=[20, 10], batch_size=len(paper_test_idx), shuffle=True) #num_workers=12)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')



brgcn = BRGCN(in_channels=128, hidden_channels=args.hidden_channels, out_channels=dataset.num_classes,
              num_layers=args.num_layers, dropout=args.dropout, neg_slope=args.neg_slope, heads=args.heads,
              num_relations=len(data.edge_index_dict.keys()), num_nodes_dict=num_nodes_dict,
              x_types=list(x_dict.keys())).to(device)

# Create global label vector.
y_global = node_type.new_full((node_type.size(0), 1), -1)
y_global[local2global['paper']] = data.y_dict['paper']

# transfer the processed data to the device (GPU)
x_dict = {k: v.to(device) for k, v in x_dict.items()}
edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
edge_type = edge_type.to(device)
node_type = node_type.to(device)
local_node_idx = local_node_idx.to(device)
y_global = y_global.to(device)
edge_index = edge_index.to(device)

def train(epoch):
    brgcn.train()
    progress_bar = tqdm(total=paper_train_idx.size(0))
    progress_bar.set_description(f'Epoch {epoch}')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        optimizer.zero_grad()

        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = brgcn(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y = y_global[n_id][:batch_size].squeeze()
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss * batch_size
        progress_bar.update(batch_size)
    progress_bar.close()
    loss = total_loss/paper_train_idx.size(0)
    return loss

@torch.no_grad()
def test_group():
    brgcn.eval()
    out = brgcn.group_inference(x_dict, edge_index_dict, key2int)
    out = out[key2int['paper']]

    y_pred = out.argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[paper_train_idx],
        'y_pred': y_pred[paper_train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[paper_valid_idx],
        'y_pred': y_pred[paper_valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[paper_test_idx],
        'y_pred': y_pred[paper_test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def test_sample():
    brgcn.eval()
    y_pred = data.y_dict['paper'].new_full((num_nodes_dict[key2int['paper']], 1), -1)
    for batch_size, n_id, adjs in train_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = brgcn(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y_pred[local_node_idx[n_id[:batch_size]]] = out.argmax(dim=-1, keepdim=True).cpu()

    for batch_size, n_id, adjs in valid_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = brgcn(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y_pred[local_node_idx[n_id[:batch_size]]] = out.argmax(dim=-1, keepdim=True).cpu()

    for batch_size, n_id, adjs in test_loader:
        n_id = n_id.to(device)
        adjs = [adj.to(device) for adj in adjs]
        out = brgcn(n_id, x_dict, adjs, edge_type, node_type, local_node_idx)
        y_pred[local_node_idx[n_id[:batch_size]]] = out.argmax(dim=-1, keepdim=True).cpu()

    y_true = data.y_dict['paper']
    train_acc = evaluator.eval({
        'y_true': y_true[paper_train_idx],
        'y_pred': y_pred[paper_train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[paper_valid_idx],
        'y_pred': y_pred[paper_valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[paper_test_idx],
        'y_pred': y_pred[paper_test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc

@torch.no_grad()
def test_fullbatch():
    brgcn.eval()

    out = brgcn.fullBatch_inference(x_dict, edge_index, edge_type, node_type, local_node_idx)
    y_pred = out[local2global['paper']].argmax(dim=-1, keepdim=True).cpu()
    y_true = data.y_dict['paper']

    train_acc = evaluator.eval({
        'y_true': y_true[paper_train_idx],
        'y_pred': y_pred[paper_train_idx],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[paper_valid_idx],
        'y_pred': y_pred[paper_valid_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[paper_test_idx],
        'y_pred': y_pred[paper_test_idx],
    })['acc']

    return train_acc, valid_acc, test_acc

test_group()
test_sample()

for run in range(args.runs):
    optimizer = torch.optim.Adam(brgcn.parameters(), lr=args.lr)
    brgcn.reset_parameters()
    for epoch in range(1, args.epochs+1):
        train_loss = train(epoch)
        print(f"Run {run}, Training epoch {epoch},  loss: {train_loss: .4f}")
        # train_acc, valid_acc, test_acc = test_fullbatch()
        train_acc, valid_acc, test_acc = test_sample()
        print(f"Testing performance sampling, "
              f"train: {100*train_acc: .2f}%,"
              f"valid: {100*valid_acc: .2f}%,"
              f"test: {100*test_acc: .2f}%")
        train_acc2, valid_acc2, test_acc2 = test_group()
        print(f"Testing performance group, "
              f"train: {100*train_acc2: .2f}%,"
              f"valid: {100*valid_acc2: .2f}%,"
              f"test: {100*test_acc2: .2f}%")