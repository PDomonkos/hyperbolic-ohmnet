# Code is adapted from https://github.com/mims-harvard/ohmnet
# MIT License
# Copyright (c) 2017 Marinka Zitnik

# Data is available at https://snap.stanford.edu/ohmnet/

import os
import re

import numpy as np

import networkx as nx


def read_tissue_hierarchy(data_path):
    hierarchy_path = os.path.join(data_path, "bio-tissue-hierarchy", "tissue.edges")
    
    tissue_hierarchy_tree = nx.DiGraph()
    with open(hierarchy_path) as fin:
        for line in fin:
            c, p = line.strip().split()
            p = re.sub(r'(?<!^)(?=[A-Z])', '_', p).lower().replace('-', '').replace('+', '').replace(':', '')
            c = re.sub(r'(?<!^)(?=[A-Z])', '_', c).lower().replace('-', '').replace('+', '').replace(':', '')
            tissue_hierarchy_tree.add_edge(p, c)
    
    assert nx.is_tree(tissue_hierarchy_tree), 'Hierarchy should be a tree'

    print("\tTissue hierarchy with " + str(len(tissue_hierarchy_tree.nodes())) + " nodes.")

    return tissue_hierarchy_tree


def read_ppi_networks(data_path, tissue_hierarchy_tree):
    networks_dir = os.path.join(data_path, "bio-tissue-networks")

    ppi_networks = dict()
    num_nodes = []
    num_edges = []
    for file in os.listdir(networks_dir): 
        tissue = file.split(".")[0]
        assert tissue in tissue_hierarchy_tree.nodes()

        tissue_ppi = nx.read_edgelist(os.path.join(networks_dir, file), nodetype=int)

        def relabel_nodes(x): return '%s__%d' % (tissue, x)
        tissue_ppi = nx.relabel_nodes(tissue_ppi, relabel_nodes)
        ppi_networks[tissue] = tissue_ppi

        num_nodes.append(tissue_ppi.number_of_nodes())
        num_edges.append(tissue_ppi.number_of_edges())
    
    print("\t" + str(len(num_nodes)) + " tissue-specific PPI networks, with an average of " + str(round(np.array(num_nodes).mean(),2)) + " nodes and " + str(round(np.array(num_edges).mean(),2)) + " edges.")

    return ppi_networks


def read_labels(data_path, tissue_hierarchy_tree):
    labels_dir = os.path.join(data_path, "bio-tissue-labels")

    ppi_labels = dict()
    num_tissues_with_labels = 0
    num_total_cellular_functions = 0
    unique_cellular_functions = []
    for file in os.listdir(labels_dir): 
        if file.endswith(".lab"):
            tissue = file.split("_GO_")[0]
            assert tissue in tissue_hierarchy_tree.nodes()

            if tissue not in ppi_labels:
                ppi_labels[tissue] = dict()
                num_tissues_with_labels += 1
            cellular_function = ("GO_" + file.split("_GO_")[1]).split(".")[0]
            if cellular_function not in unique_cellular_functions:
                unique_cellular_functions.append(cellular_function)
            num_total_cellular_functions += 1
            ppi_labels[tissue][cellular_function] = dict()

            with open(os.path.join(labels_dir, file)) as fin:
                for line in fin:
                    if line[0] == "#":
                        continue
                    protein, label = line.strip().split()
                    ppi_labels[tissue][cellular_function][protein] = label
    
    print("\t" + str(num_total_cellular_functions) + " tissue-specific cellular functions (with " + str(len(unique_cellular_functions)) + " unique functions) covering " + str(num_tissues_with_labels) + " distinct tissues.")

    return ppi_labels


def read_data(data_path):
    print("Read data:")

    tissue_hierarchy_tree = read_tissue_hierarchy(data_path)
    ppi_networks = read_ppi_networks(data_path, tissue_hierarchy_tree)
    ppi_labels = read_labels(data_path, tissue_hierarchy_tree)

    return tissue_hierarchy_tree, ppi_networks, ppi_labels