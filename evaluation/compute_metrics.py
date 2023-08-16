from functools import partial
import numpy as np
import networkx as nx
import os
import pickle as pkl
import subprocess as sp
import time


motif_to_indices = {
        '3path' : [1, 2],
        '4cycle' : [8],
}
COUNT_START_STR = 'orbit counts: \n'


def get_clustering(graphs):
    val = []
    for g in graphs:
        clustering_coeffs_list = list(nx.clustering(g).values())
        hist, _ = np.histogram(clustering_coeffs_list, bins=100, range=(0.0, 1.0), density=False)
        val.append(hist)
    return val
def get_degs(graphs):
    val = []
    for g in graphs:
        val.append(nx.degree_histogram(g))
    return val

def get_clos(graphs):
    val = []
    for g in graphs:
    	a,_ = np.histogram(list(nx.closeness_centrality(g).values()))
    	val.append(list(a))
    return val
    
def get_asp(graphs):
    val = []
    for g in graphs:
    	Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    	G0 = g.subgraph(Gcc[0])
    	a = nx.average_shortest_path_length(G0)
    	val.append(a)
    	
    a,_ = np.histogram(val)
    return [a]
   
    
def get_eccen(graphs):
    val = []
    for g in graphs:
    	Gcc = sorted(nx.connected_components(g), key=len, reverse=True)
    	G0 = g.subgraph(Gcc[0])
    	a,_ = np.histogram(list(nx.eccentricity(G0).values()))
    	val.append(list(a))
    return val

def get_neigh_deg(graphs):
    val = []
    for g in graphs:
    	a,_ = np.histogram(list(nx.assortativity.average_neighbor_degree(g).values()))
    	val.append(list(a))
    return val
    
def get_betw(graphs):
    val = []
    for g in graphs:
    	a,_ = np.histogram(list(nx.betweenness_centrality(g).values()))
    	val.append(list(a))
    return val


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_fname = 'orca/tmp.txt'
    f = open(tmp_fname, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    #output = sp.check_output(['orca/', 'node', '4', 'orca/tmp.txt', 'std'])
    output = sp.check_output(['orca/orca', 'node', '4', 'orca/tmp.txt', 'std'])
    output = output.decode('utf8').strip()
    
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ') ))
          for node_cnts in output.strip('\n').split('\n')])

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts
    
def get_orbit(graphs):
    total_counts = []
    graphs_remove_empty = [G for G in graphs if not G.number_of_nodes() == 0]
    
    for G in graphs_remove_empty:
        try:
            orbit_counts = orca(G)
        except:
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts.append(orbit_counts_graph)
        
    return total_counts
