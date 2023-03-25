import torch
from enum import IntEnum
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


class Electrode(IntEnum):
    EEG_F3 = 0
    EEG_C3 = 1
    EEG_O1 = 2
    EEG_F4 = 3
    EEG_C4 = 4
    EEG_O2 = 5
    EOG_Right = 6
    EOG_Left = 7
    EMG_Chin = 8
    ECG = 9
    EMG_Leg = 10

def _get_edge_type(node1, node2):
    i = _get_node_type(node1.item())
    j = _get_node_type(node2.item())
    return i * 4 + j


def _get_node_type(node):
    if Electrode(node).name[:3] == 'EEG':
        return 0
    elif Electrode(node).name[:3] == 'EOG':
        return 1
    elif Electrode(node).name[:3] == 'EMG':
        return 2
    elif Electrode(node).name[:3] == 'ECG':
        return 3


def get_batches(cfg, data, label, edge_index, fold):
    all_graphs = []
    batched_train_graphs = []
    batched_val_graphs = []

    n_subjects = cfg.const.n_subjects

    for subject in range(n_subjects):
        node_type = []
        edge_type = []
        for i in range(data[subject].shape[1]):
            node_type.append([_get_node_type(i)])
        for i, j in edge_index[subject].t():
            edge_type.append([_get_edge_type(i, j)])
        node_type = torch.tensor(node_type, dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)

        subject_graphs = []
        for i in range(data[subject].shape[0]):
            d = Data(x=data[subject][i], edge_index=edge_index[subject], y=label[subject][i].view(-1, 1))
            d.node_type = node_type
            d.edge_type = edge_type
            subject_graphs.append(d)
        all_graphs.append(subject_graphs)

    n_subject_in_fold = n_subjects // cfg.k_fold
    train_graphs = []
    for train_subjects in all_graphs[:fold * n_subject_in_fold] + all_graphs[(fold + 1) * n_subject_in_fold:]:
        for graph in train_subjects:
            train_graphs.append(graph)
    val_graphs = []
    for val_subjects in all_graphs[fold * n_subject_in_fold: (fold + 1) * n_subject_in_fold]:
        for graph in val_subjects:
            val_graphs.append(graph)

    batch_size = cfg.batch_size
    i = 0
    while i * batch_size < len(train_graphs):
        print(f'train batch {i + 1}...')
        start = i * batch_size
        end = min(start + batch_size, len(train_graphs))
        train_batch = Batch.from_data_list(train_graphs[start: end])
        batched_train_graphs.append(train_batch)
        i += 1

    i = 0
    while i * batch_size < len(val_graphs):
        print(f'val batch {i + 1}...')
        start = i * batch_size
        end = min(start + batch_size, len(val_graphs))
        val_batch = Batch.from_data_list(val_graphs[start: end])
        batched_val_graphs.append(val_batch)
        i += 1

    return batched_train_graphs, batched_val_graphs
