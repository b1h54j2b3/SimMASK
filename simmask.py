# encoding=utf-8
import os
import copy
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import logging
from copy import deepcopy

from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
from gin import Encoder
from evaluate_embedding import evaluate_embedding

if torch.cuda.device_count() > 0:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

save_path = os.getcwd()
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, num_features, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch):
        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch)
        y = self.proj_head(y)
        return y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def loss_cal2(self, x, x_aug):

        batch_size, _ = x.size()

        x_abs = x.norm(dim=1, p=2)
        x_aug_abs = x_aug.norm(dim=1, p=2)

        x = torch.div(x, torch.repeat_interleave(x_abs.unsqueeze(-1), x.shape[1], dim=1))
        x_aug = torch.div(x_aug, torch.repeat_interleave(x_aug_abs.unsqueeze(-1), x_aug.shape[1], dim=1))

        sim = torch.matmul(x, x_aug.t())

        loss = - sim.mean()

        return loss


def gen_ran_output(data, model, vice_model, init_model, p, beta=0.95):
    for (_, adv_param), (name, param), (_, init_param) in zip(vice_model.named_parameters(), model.named_parameters(),
                                                              init_model.named_parameters()):
        if name.split('.')[0] == 'proj_head':
            adv_param.data = param.data
        else:
            mask = torch.ones_like(param.data).uniform_(0, 1)
            adv_param.data[mask >= p] = beta * adv_param.data[mask >= p] + (1.0 - beta) * param.data[mask >= p]
            adv_param.data[mask < p] = init_param.data[mask < p]
    z2 = vice_model(data.x, data.edge_index, data.batch)
    return z2, vice_model



def train_model(lr, beta, p, dataset_name, layers, seed, epochs, batch_size, hidden_dim=32):
    setup_seed(seed)

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '.', 'data', dataset_name)
    dataset = TUDataset(path, name=dataset_name).shuffle()
    dataset_eval = TUDataset(path, name=dataset_name).shuffle()

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    model = simclr(hidden_dim, layers, dataset_num_features).to(device)
    vice_model = simclr(hidden_dim, layers, dataset_num_features).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print('================')
    print('dataset: {}'.format(dataset_name))
    print('train num: {}, eval num:{}'.format(len(dataset), len(dataset_eval)))
    print('lr: {}'.format(lr))
    print('beta: {}'.format(beta))
    print('p: {}'.format(p))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(hidden_dim))
    print('num_gc_layers: {}'.format(layers))
    print('batch_size: {}'.format(batch_size))
    print('epochs: {}'.format(epochs))
    print('================')

    model.eval()
    vice_model.eval()
    init_model = copy.deepcopy(vice_model)

    for epoch in range(1, epochs + 1):

        loss_all = 0
        model.train()

        for data in dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            x2, vice_model = gen_ran_output(data, model, vice_model, init_model, p, beta=beta)
            x1 = model(data.x, data.edge_index, data.batch)
            x2 = Variable(x2.detach().data, requires_grad=False)

            loss_aug = model.loss_cal(x2, x1)
            loss = loss_aug
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()

        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    acc_val, acc = evaluate_embedding(emb, y)
    print('acc:' + str(acc) + ' acc val:' + str(acc_val))

    return acc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='simmask')
    parser.add_argument('--dataset', default='MUTAG')
    parser.add_argument('--beta', default=-1)
    parser.add_argument('--p', default=-1)
    args = parser.parse_args()

    import itertools

    if args.beta == -1:
        betalist = [0.99, 0.95, 0.9, 0.0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    else:
        betalist = [float(args.beta)]

    if args.p == -1:
        plist = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    else:
        plist = [float(args.p)]

    # betalist = [0.9]
    # plist = [0.7]
    # lrlist = [0.001, 0.01, 0.1]
    lrlist=[0.01]
    # layerlist = [2, 3, 4, 5, 6]
    layerlist = [3]
    seedlist = [0, 1, 2, 3, 4]

    batch_size = 128
    epochs = 20
    dataset_name = args.dataset

    LOG_FORMAT = "%(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"

    logging.basicConfig(filename='Accuracy_' + dataset_name + '.txt', level=logging.DEBUG,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT)

    params = []
    for beta, lr, p, layer in itertools.product(betalist, lrlist, plist, layerlist):
        params.append([beta, lr, p, layer])

    for beta, lr, p, layer in params:
        accs = []
        for seed in seedlist:
            acc = train_model(lr, beta, p, dataset_name, layer, seed, epochs, batch_size, hidden_dim=32)
            accs.append(acc)

        accs.append(sum(accs) / len(accs))
        accs = [str(e) for e in accs]

        logging.info('model: beta=' + str(beta) + ' lr=' + str(lr) + ' p=' + str(p) + ' layer=' + str(layer))
        logging.info('acc: ' + ','.join(accs))
