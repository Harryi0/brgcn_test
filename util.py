import torch

def process_mag(data):
    '''
    :param data: mag dataset
    :return:
    all_edge_index: all of the edges index from source to  target
    edge_type: edge type (relation) for each edge in the all_edge_index
    node_type: node type for each node
    global2local: global index to local index of each node
    local2global: local index to global index of each node
    node_relation2int: node type and relation type to their index
    '''
    edge_index_dict = data.edge_index_dict
    num_nodes_dict = data.num_nodes_dict

    e_example = list(edge_index_dict.values())[0]
    node_relation2int, node2offset, offset = {}, {}, 0
    node_type = []
    global2local, local2global = [], {}
    for i, (n, num) in enumerate(num_nodes_dict.items()):
        node_relation2int[n] = i
        global2local.append(torch.arange(num, device=e_example.device))
        local2global[n] = global2local[-1] + offset
        node_type.append(e_example.new_full((num,), i))
        node2offset[n] = offset
        offset += num

    all_edge_index, edge_type  = [], []
    relation_idx = 0
    for relation, edge_index in edge_index_dict.items():
        n1, n2 = relation[0], relation[2]
        reverse_realtion = (n2, 'reverse', n1)
        node_relation2int[relation] = relation_idx
        all_edge_index.append(edge_index + torch.tensor([offset[n1], offset[n2]], device=edge_index.device))
        edge_type.append(edge_index.new_full((edge_index.size(1),), relation_idx))
        relation_idx += 1

        node_relation2int[reverse_realtion] = relation_idx
        reverse_edge_index = edge_index[[1, 0], :]
        all_edge_index.append(reverse_edge_index + torch.tensor([offset[n2], offset[n1]], device=edge_index.device))
        edge_type.append(reverse_edge_index.new_full((reverse_edge_index.size(1),), relation_idx))
        relation_idx += 1

    global2local = torch.cat(global2local, dim=0)
    node_type = torch.cat(node_type, dim=0)
    all_edge_index = torch.cat(all_edge_index, dim=1)
    edge_type = torch.cat(edge_type, dim=0)

    return all_edge_index, edge_type, node_type, global2local, local2global, node_relation2int