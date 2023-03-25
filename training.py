import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from config import *

from data import get_batches
from util import load_data, plot_Matrix
from model import MyModel

import torch
import torch.nn.functional as F
from torch.optim import Adam

from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
import os

class Trainer:
    def __init__(self, cfg: BaseConfig):
        super().__init__()

        self.cfg = cfg

        self.all_data, self.all_label, self.all_edge_index = load_data(self.cfg)

        in_dim = self.all_data[0].shape[-1]
        out_dim = 5
        n_node_in_graph = self.all_data[0].shape[-2]

        self.training_accs, self.training_losses = [], []
        self.val_accs, self.val_losses = [], []
        self.y, self.y_pred = [], []

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = MyModel(
                in_dim, self.cfg.emb_dim, out_dim, 
                self.cfg.const.n_node_types, self.cfg.const.n_relation_types, 
                n_node_in_graph=n_node_in_graph,
                n_HGTs=self.cfg.n_HGTs, n_heads=self.cfg.n_heads, 
                lin_dims=self.cfg.lin_dims, lin_dropout=self.cfg.lin_dropout
            )
        self.model.to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.l2_decay)
        self.model.show_parameter_num()

        output_root = self.cfg.output_root
        assert not os.path.exists(output_root), 'output_root already exists, files would be overwrited.'
        os.mkdir(output_root)

        os.mkdir(self.cfg.criterion_root)
        os.mkdir(self.cfg.plot_root)


    def run(self, train_graph, val_graph, fold, max_epochs):
        fold_training_accs, fold_training_losses = [], []
        fold_val_accs, fold_val_losses = [], []
        fold_ys, fold_y_preds = [], []

        for epoch in range(1, max_epochs + 1):
            print(f'epoch{epoch}...')
            t_start = time.perf_counter()

            fold_training_acc, fold_training_loss = self.train(train_graph)
            fold_val_acc, fold_val_loss, fold_y, fold_y_pred = self.evaluate(val_graph)
            t_end = time.perf_counter()
            print(f'epoch{epoch} time: {t_end - t_start:.2f}s\tacc: {fold_training_acc:.3f}\tloss: {fold_training_loss:.3f}'
                + f'\tval_acc: {fold_val_acc:.3f}\tval_loss: {fold_val_loss:.3f}')

            fold_training_accs.append(fold_training_acc)
            fold_training_losses.append(fold_training_loss)
            fold_val_accs.append(fold_val_acc)
            fold_val_losses.append(fold_val_loss)
            fold_ys.append(fold_y)
            fold_y_preds.append(fold_y_pred)

            print('=' * 20)

        self.training_accs.append(fold_training_accs)
        self.training_losses.append(fold_training_losses)
        self.val_accs.append(fold_val_accs)
        self.val_losses.append(fold_val_losses)
        best_idx = torch.tensor(fold_val_accs).max(0)[1].item()
        self.y += fold_ys[best_idx]
        self.y_pred += fold_y_preds[best_idx]

        names = ['acc', 'loss', 'val_acc', 'val_loss']
        values = [fold_training_accs, fold_training_losses, fold_val_accs, fold_val_losses]
        for name, value in zip(names, values):
            with open(os.path.join(self.cfg.criterion_root, f'fold{fold + 1}_{name}'), 'wb') as f:
                pickle.dump(value, f)


    def train(self, train_graph):
        self.model.train()

        training_acc, training_loss = 0, 0
        n_graph = 0

        n_batch = len(train_graph)
        n_graph += (n_batch - 1) * self.cfg.batch_size + train_graph[-1].num_graphs
        for j in range(n_batch):
            graph = train_graph[j].to(self.device)

            self.optimizer.zero_grad()
            out = self.model(graph).unsqueeze(-1)
            acc = out.max(1)[1].eq(graph.y).sum().item()
            loss = F.nll_loss(out, graph.y, reduction='sum')
            loss_num = loss.detach().item()
            training_acc += acc
            training_loss += loss_num

            loss.backward()
            self.optimizer.step()

        print('-' * 20)

        training_acc /= n_graph
        training_loss /= n_graph

        return training_acc, training_loss


    def evaluate(self, val_graph):
        self.model.eval()

        all_acc = 0
        all_loss = 0
        n_graph = 0
        batch_y, batch_y_pred = [], []

        n_batch = len(val_graph)
        n_graph += (n_batch - 1) * self.cfg.batch_size + val_graph[-1].num_graphs
        for j in range(n_batch):
            with torch.no_grad():
                graph = val_graph[j].to(self.device)
                logits = self.model(graph).unsqueeze(-1)
                pred = logits.max(1)[1]

            l = graph.y
            loss = F.nll_loss(logits, l, reduction='sum')
            all_acc += pred.eq(l).sum().item()
            all_loss += loss.detach().item()
            batch_y += l.squeeze().tolist()
            batch_y_pred += pred.squeeze().tolist()

        epoch_acc = all_acc / n_graph
        epoch_loss = all_loss / n_graph

        return epoch_acc, epoch_loss, batch_y, batch_y_pred
    
    def save_overall_figures(self):
        training_accs = np.mean(self.training_accs, axis=0)
        training_losses = np.mean(self.training_losses, axis=0)
        val_accs = np.mean(self.val_accs, axis=0)
        val_losses = np.mean(self.val_losses, axis=0)

        epochs = range(1, len(training_losses) + 1)
        plt.plot(epochs, training_accs, 'bo', label='Training acc')
        plt.plot(epochs, val_accs, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.plot_root, f'acc.jpg'))

        plt.figure()
        plt.plot(epochs, training_losses, 'bo', label='Training loss')
        plt.plot(epochs, val_losses, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig(os.path.join(self.cfg.plot_root, f'loss.jpg'))

        plt.figure()
        plot_Matrix(self.cfg, self.y, self.y_pred)

        precision_recall_f1 = precision_recall_fscore_support(self.y, self.y_pred, average='macro')
        precision_recall_f1_file = open(os.path.join(self.cfg.criterion_root, f'precision_recall_f1'), 'wb')
        pickle.dump(precision_recall_f1, precision_recall_f1_file)
        precision_recall_f1_file.close()

        y_file = open(os.path.join(self.cfg.criterion_root, f'y'), 'wb')
        y_pred_file = open(os.path.join(self.cfg.criterion_root, f'y_pred'), 'wb')
        pickle.dump(self.y, y_file)
        pickle.dump(self.y_pred, y_pred_file)
        y_file.close()
        y_pred_file.close()
    

    def start(self):
        for i in range(self.cfg.k_fold):
            print(f'=====fold {i + 1}=====')
            self.model.reset_parameters()
            
            batched_train_graph, batched_val_graph = get_batches(self.cfg, self.all_data, self.all_label, self.all_edge_index, fold=i)
            self.run(batched_train_graph, batched_val_graph, fold=i, max_epochs=self.cfg.max_epochs)

            del batched_train_graph, batched_val_graph
            torch.cuda.empty_cache()

        self.save_overall_figures()


if __name__ == '__main__':
    cs = ConfigStore.instance()
    cs.store(name="config", node=MyConfig)

    @hydra.main(version_base=None, config_name="config")
    def my_app(cfg: BaseConfig):
        trainer = Trainer(cfg)
        trainer.start()

    my_app()

    