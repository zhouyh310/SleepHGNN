from data import Electrode
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def load_data(cfg):
    all_data = []
    all_label = []
    all_edge_index = []

    for subject in range(cfg.const.n_subjects):
        data_path = os.path.join(cfg.data_root, cfg.feature_dirname, f'subject{subject + 1}.npy')
        label_path = os.path.join(cfg.data_root, cfg.label_dirname, f'{subject + 1}_1.npy')

        subject_data = np.load(data_path)
        subject_label = np.load(label_path)

        if cfg.shuffle:
            perm = np.random.permutation(range(len(subject_label)))
            subject_data = subject_data[perm]
            subject_label = subject_label[perm]

        all_data.append(torch.tensor(subject_data, dtype=torch.float))
        all_label.append(torch.tensor(subject_label, dtype=torch.long).view(-1, 1))

        adj_mat_path = os.path.join(cfg.data_root, cfg.adj_mat_dirname, f'subject_{subject + 1}_adj_mat.npy')
        subject_adj_mat = np.load(adj_mat_path)
        subject_adj_mat = torch.tensor(subject_adj_mat)
        edge_pair, _ = dense_to_sparse(subject_adj_mat)
        subject_edges = []
        for j in range(len(edge_pair[0])):
            subject_edges.append([Electrode(edge_pair[0][j].item()), Electrode(edge_pair[1][j].item())])

        subject_edge_index = []
        for edge_pair in subject_edges:
            subject_edge_index += [[edge_pair[0].value, edge_pair[1].value]]
            subject_edge_index += [[edge_pair[1].value, edge_pair[0].value]]
        for i in range(len(Electrode)):
            subject_edge_index += [[Electrode(i).value, Electrode(i).value]]
            if 'EEG' not in Electrode(i).name:
                for j in range(i + 1, len(Electrode)):
                    subject_edge_index += [[Electrode(i).value, Electrode(j).value]]
                    subject_edge_index += [[Electrode(j).value, Electrode(i).value]]

        subject_edge_index = torch.LongTensor(subject_edge_index)
        subject_edge_index = subject_edge_index.t()
        all_edge_index.append(subject_edge_index)
        print(f'subject{subject + 1} loaded...')

    return all_data, all_label, all_edge_index


def count_parameters(model, name):
    total_params = 0
    for n, p in model.named_parameters(prefix=name):
        cur_params = p.numel()
        total_params += cur_params
    print(f'{name} {total_params:,} parameters.')
    print('----------------------------------------')
    return total_params


def plot_Matrix(cfg, y, yp):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yp)
    np.save(os.path.join(cfg.criterion_root, f'confusion_matrix.npy'), cm)
    cm = cm.astype('float32')
    for i in range(len(cm)):
        cm[i] /= np.sum(cm[i])
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(f'{cm[x, y]:.3f}', xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label') 
    plt.xlabel('Predicted label') 
    plt.savefig(os.path.join(cfg.plot_root, f'confusion_matrix.jpg'))
