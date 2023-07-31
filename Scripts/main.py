import numpy as np
import networkx as nx
import math
import time
import pickle as pkl
import scipy
import argparse
import random
import sys
sys.path.insert(0, '/home/azzolin/AHK') # TODO: change

import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
from torch_geometric.data import InMemoryDataset, download_url

import utils
from ahk import AHK_graphon
from ahk_generators import data_colors, ahk_sbm, sample_data

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

import mmd_digress
from compute_metrics import get_orbit, get_clustering, get_degs


class GraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']

    def download(self):
        """
        Download raw files as originally defined in the SPECTRE repo
        """
        if self.dataset_name == 'sbm':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.dataset_name == 'planar':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.dataset_name == 'comm20':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        file_path = download_url(raw_url, self.raw_dir)

        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])


    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = None #torch.ones(n, 1, dtype=torch.float)
            y = None #torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = None #torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            # edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


def dataset_to_nx(dataset):
        networkx_graphs = []
        for i, data in enumerate(dataset):
            # data_list = batch.to_data_list()
            # for j, data in enumerate(data_list):
            networkx_graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True,
                                                remove_self_loops=True))
        return networkx_graphs

def get_graphs(name):
    if name in ["sbm", "planar"]:
        test_reference_graphs = dataset_to_nx(GraphDataset(name, split="test", root=f"/home/azzolin/data/{name}"))
        train_reference_graphs = dataset_to_nx(GraphDataset(name, split="train", root=f"/home/azzolin/data/{name}"))
        val_reference_graphs = dataset_to_nx(GraphDataset(name, split="val", root=f"/home/azzolin/data/{name}"))
    elif name == "grid":
        print("TO BE FINISHED")
        graphs = []
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))
        random.seed(123)
        random.shuffle(graphs)
        graphs_len = len(graphs)
        test_reference_graphs = graphs[int(0.8 * graphs_len):]
        train_reference_graphs = graphs[0:int(0.8*graphs_len)]
    return train_reference_graphs, val_reference_graphs, test_reference_graphs


def convert_nx_to_world(datasets):
    map = {
        "features": {
            0: 0
        }
    }
    ret = []
    for graphs in datasets:
        tmp = []
        for g in graphs:
            nx.set_node_attributes(g, 0, "features") # dummy node attribute required by AHK (TODO: fix)
            w = utils.nx_to_world(g, featmaps=map)
            tmp.append(w)
        ret.append(tmp)
    return ret

def train(config, model, graphs_train, graphs_val, graphs_test):
    best, loglik, trace = model.learn(config, graphs_train)

def eval(generated, reference):
    print(f"Evaluating {len(generated)} generated graphs")

    s1 = get_degs(reference)
    s2 = get_degs(generated)
    degs = mmd_digress.compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False)
    print("Degree stat: ", degs)

    s1 = get_orbit(reference)
    s2 = get_orbit(generated)
    orbit = mmd_digress.compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, is_hist=False, sigma=30.0)
    print("Orbit stat: ", orbit)

    s1 = get_clustering(reference)
    s2 = get_clustering(generated)
    clust = mmd_digress.compute_mmd(s1, s2, kernel=mmd_digress.gaussian_tv, is_parallel=False, sigma=1.0 / 10)
    print("Clustering stat: ", clust)


def main(args, config):
    print(f"Training on {args.dataset_name}...")
    print(config)

    graphs_train_nx, graphs_val_nx, graphs_test_nx = get_graphs(args.dataset_name)
    print(f"Num train graphs = {len(graphs_train_nx)}, Num test graphs = {len(graphs_test_nx)}")
    print(f"Avg num nodes train graphs = ", np.mean([len(g.nodes()) for g in graphs_train_nx]))
    print(f"Min/Max num nodes train graphs = ", np.min([len(g.nodes()) for g in graphs_train_nx]), np.max([len(g.nodes()) for g in graphs_train_nx]))
    print()

    graphs_train, graphs_val, graphs_test = convert_nx_to_world([graphs_train_nx, graphs_val_nx, graphs_test_nx])

    # Init model
    binbounds = utils.uni_bins(3)  # HOW TO SET THIS PARAM?
    model = AHK_graphon(graphs_train[0].sig, binbounds)
    model.rand_init()

    train(config, model, graphs_train[:20], graphs_val, graphs_test)

    generated_graphs = sample_data(model, n=10, minnodes=5, maxnodes=30)
    generated_graphs_nx = [nx.Graph(g.to_nx()) for g in generated_graphs]

    eval(generated_graphs_nx, graphs_test_nx)



if __name__ == '__main__':
    """
    Example usage (background execution):
        nohup python main.py --dataset_name planar > planar.out 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='sbm', help='dataset name to train on')
    args = parser.parse_args()

    config = {}  # TODO: define in external file
    config['num_pi_b'] = 50
    config['batchsize'] = 10
    config['soft'] = 0.01
    config['numepochs'] = 2
    config['learn_bins'] = False
    config['early_stop'] = 5
    config['with_trace'] = False

    # Adam params:
    config['ad_alpha'] = 0.01
    config['ad_beta1'] = 0.99
    config['ad_beta2'] = 0.9
    config['ad_epsilon'] = 10e-8
    config['method'] = "adam"

    main(args, config)
