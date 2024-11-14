import os

import numpy as np 
import pandas as pd

import networkx as nx

from tqdm.notebook import tqdm

import multiprocessing

from utils.hyperbolic_utils import poincare_distance, lorentz_to_poincare, euclidean_distance, lorentzian_distance_matrix

from utils.tissue_embedding import LeafTissueEmbedding, InternalTissueEmbedding

from matplotlib import pyplot as plt
import seaborn as sns



class TissueSpecificProteinEmbeddings():
    """
        Tissue-specific PPI graph embeddings.
        Pytorch implementation of the OhmNet model (https://github.com/mims-harvard/ohmnet) operating with different manifolds.

        Attributes
        ----------
        tissue_hierarchy: nx.DiGraph
            Tissue hierarchy tree.
        leaf_tissues: list
            PPI embedding modules corresponding to leaf nodes in the hierarchy.
        internal_tissues: list
            PPI embedding modules corresponding to internal nodes in the hierarchy.
        config: dict
            Hyperparameter configuration.
        config_name: str
            Name of the model based on the configuration.
        tissue_specific_embeddings: dict
            A dictionary storing the embedding modules for all tissue nodes.
        
        Methods
        -------
        init(tissue_hierarchy, ppi_networks, labels, config, eval)
            Instantiate the leaf and internal embedding modules.
            If "eval" is True, store the necessary data (labels, graph distances) for the later evaluation.
        initialize_embeddings(ppi_networks)
            Initialize the leaf and internal embeddings.
        add_labels_for_classification(labels)
            Set the GO labels for node classification.
        add_distances_for_distortion(ppi_networks)
            Set the shortest path distances for distortion evaluation.
        train()
            Train the whole model based on the configuration according to the OhmNet algorithm.
        save(path)
            Save the stored results and the embeddings to the given path.
        load(path)
            Load and store previous results and the embeddings from the given path.
        get_embeddings(): dict
            Return the learned embeddings and corresponding entrez IDs for each tissue.
        evaluate_embeddings(save, classification, distortion): dict
            Evaluate the embeddings and return the results of the node classification (if the "classification" parameter is True) and the graph distortion (if the "distortion" parameter is True) tasks.
        evaluate_embeddings_during_training(loop)
            Evaluate the embeddings, store and log the results.
        show_classification_results()
            Perform an evaluation and plot the classification results.
        show_distortion_results()
            Perform an evaluation and plot the graph distortion results.
        show_results_over_epochs()
            Plot all saved results over epochs.
        show_classification_results_over_epochs()
            Plot classification metrics over epochs.
        show_distortion_results_over_epochs()
            Plot graph distortion metrics over epochs.
    """

    def __init__(self, tissue_hierarchy, ppi_networks, labels, config, eval = True):
        self.__tissue_hierarchy = tissue_hierarchy
        self.__leaf_tissues = [node for node in self.__tissue_hierarchy.nodes() if self.__tissue_hierarchy.out_degree()[node]==0]
        self.__internal_tissues = [node for node in list(reversed(list(nx.topological_sort(self.__tissue_hierarchy)))) if node not in self.__leaf_tissues]
        self.__config = config
        self.__config_name = '_'.join(key + str(val).replace(".", "").replace(':', '').split(" ")[0] for key, val in config.items())

        self.__initialize_embeddings(ppi_networks)
        if eval:
            self.__add_labels_for_classification(labels)
            self.__add_distances_for_distortion(ppi_networks)

    def __initialize_embeddings(self, ppi_networks):
        self.__tissue_specific_embeddings = dict()
        for tissue in self.__leaf_tissues:
            self.__tissue_specific_embeddings[tissue] = LeafTissueEmbedding(tissue, self.__config, ppi_networks[tissue])
        for tissue in self.__internal_tissues:
            children = [self.__tissue_specific_embeddings[child] for child in self.__tissue_hierarchy.successors(tissue)]
            self.__tissue_specific_embeddings[tissue] = InternalTissueEmbedding(tissue, self.__config, children) 

    def __add_labels_for_classification(self, labels):
        for tissue, model in self.__tissue_specific_embeddings.items():
            if tissue in labels:
                model.set_node_labels(labels[tissue])

    def __add_distances_for_distortion(self, ppi_networks):
        tissues = np.array(list(ppi_networks.keys()))
        largest_connected_components = [tissue_ppi.subgraph(max(nx.connected_components(tissue_ppi), key=len)) for tissue_ppi in ppi_networks.values()]
        with multiprocessing.Pool(30) as p:
            shortest_distance_matrices = p.map(nx.floyd_warshall_numpy, largest_connected_components)
        for tissue, model in self.__tissue_specific_embeddings.items():
            if tissue in tissues: 
                i = np.where(tissue == tissues)[0][0]
                distance_matrix = shortest_distance_matrices[i]
                ppi = largest_connected_components[i]

                ppi_entrez_ids = np.array([n.split("__")[1] for n in np.array(list(ppi.nodes))])
                model_entrez_ids = model.get_entrez_ids()

                protein_in_model_mask = np.isin(ppi_entrez_ids, model_entrez_ids)

                distance_matrix = distance_matrix[protein_in_model_mask,:][:,protein_in_model_mask]
                ppi_entrez_ids = ppi_entrez_ids[protein_in_model_mask]

                model.set_graph_distances(distance_matrix, ppi_entrez_ids)

    def train(self):
        loop = tqdm(range(self.__config["num_epochs"]), desc="Training OhmNet")
        if self.__config["test_epochs"] != 0:
            self.__evaluate_embeddings_during_training(loop)
        for e in loop:
            for tissue in self.__leaf_tissues:
                self.__tissue_specific_embeddings[tissue].update_embeddings()
            
            for tissue in self.__internal_tissues:
                self.__tissue_specific_embeddings[tissue].update_embeddings()

            if (self.__config["test_epochs"] != 0) and ((e+1) % self.__config["test_epochs"] == 0):
                self.__evaluate_embeddings_during_training(loop)

    def save(self, path):
        path = os.path.join(path, self.__config_name)
        if not os.path.exists(path):
            os.mkdir(path) 
        else:
            print("Overwriting directory: " + path)
        for _, model in self.__tissue_specific_embeddings.items():
            model.save(path)

    def load(self, path):
        for _, model in self.__tissue_specific_embeddings.items():
            model.load(os.path.join(path, self.__config_name))

    def get_embeddings(self):
        embedding_dict = dict()
        for tissue, model in self.__tissue_specific_embeddings.items():
            embedding_dict[tissue] = dict()
            embedding_dict[tissue]["embeds"] = model.get_embeddings().cpu().numpy()
            embedding_dict[tissue]["entrez_ids"] = model.get_entrez_ids()
        return embedding_dict

    def evaluate_embeddings(self, save = True, classification = True, distortion = True):
        results = dict()
        for tissue, model in self.__tissue_specific_embeddings.items():
            results[tissue] = model.evaluate_embeddings(save = save, classification = classification, distortion = distortion)
        return results
    
    def __evaluate_embeddings_during_training(self, loop):
        results = self.evaluate_embeddings(save = True)

        leaf_local_aurocs = []
        leaf_local_auprs = []
        leaf_global_aurocs = []
        leaf_global_auprs = []
        leaf_distortions = []
        for tissue in self.__leaf_tissues:
            if results[tissue]["node_classification"] is not None:
                for function, res in results[tissue]["node_classification"].items():
                    leaf_local_aurocs.append(res["local AUROC"])
                    leaf_local_auprs.append(res["local AUPR"])
                    leaf_global_aurocs.append(res["global AUROC"])
                    leaf_global_auprs.append(res["global AUPR"])
            if results[tissue]["embedding_distortion"] is not None:
                leaf_distortions.append(results[tissue]["embedding_distortion"])

        internal_local_aurocs = []
        internal_local_auprs = []
        internal_global_aurocs = []
        internal_global_auprs = []
        internal_distortions = []
        for tissue in self.__internal_tissues:
            if results[tissue]["node_classification"] is not None:
                for function, res in results[tissue]["node_classification"].items():
                    internal_local_aurocs.append(res["local AUROC"])
                    internal_local_auprs.append(res["local AUPR"])
                    internal_global_aurocs.append(res["global AUROC"])
                    internal_global_auprs.append(res["global AUPR"])
            if results[tissue]["embedding_distortion"] is not None:
                internal_distortions.append(results[tissue]["embedding_distortion"])

        loop.set_postfix({'leaf local AUROC': str(round(np.vstack(leaf_local_aurocs).mean(), 4)),
                        'leaf local AURPR': str(round(np.vstack(leaf_local_auprs).mean(), 4)),
                        'leaf global AUROC': str(round(np.vstack(leaf_global_aurocs).mean(), 4)),
                        'leaf global AURPR': str(round(np.vstack(leaf_global_auprs).mean(), 4)),
                        'leaf DIST': str(round(np.hstack(leaf_distortions).mean(), 4)),
                        'internal local AUROC': str(round(np.vstack(internal_local_aurocs).mean(), 4)),
                        'internal local AURPR': str(round(np.vstack(internal_local_auprs).mean(), 4)),
                        'internal global AUROC': str(round(np.vstack(internal_global_aurocs).mean(), 4)),
                        'internal global AURPR': str(round(np.vstack(internal_global_auprs).mean(), 4)),
                        'internal DIST': str(round(np.hstack(internal_distortions).mean(), 4))
                        })
    
    def show_distortion_results(self):
        distortions = dict()
        results = self.evaluate_embeddings(save = False, classification = False)
        for tissue, result in results.items():
            if result["embedding_distortion"] is not None:
                distortions[tissue] = result["embedding_distortion"]

        keys = list(distortions.keys())
        values = list(distortions.values())
        plt.figure(figsize=(6,6))
        sns.barplot(x=keys, y=values)
        plt.ylabel("Distortion (mean:" + str(round(np.array(values).mean(),4)) + ")")
        plt.xticks(fontsize=8, rotation=90)
        plt.show()

    def show_classification_results(self):
        df_local_auroc = pd.DataFrame()
        df_local_aupr = pd.DataFrame()
        df_global_auroc = pd.DataFrame()
        df_global_aupr = pd.DataFrame()
        results = self.evaluate_embeddings(save = False, distortion = False)
        for tissue, result in results.items():
            if result["node_classification"] is not None:
                n_tissue = len(result["node_classification"])
                local_aurocs = []
                local_auprs = []
                global_aurocs = []
                global_auprs = []
                for function, res in result["node_classification"].items():
                    local_aurocs.append(res["local AUROC"])
                    local_auprs.append(res["local AUPR"])
                    global_aurocs.append(res["global AUROC"])
                    global_auprs.append(res["global AUPR"])
                local_aurocs = np.hstack(local_aurocs)
                local_auprs = np.hstack(local_auprs)
                global_aurocs = np.hstack(global_aurocs)
                global_auprs = np.hstack(global_auprs)
                df_local_auroc = pd.concat([df_local_auroc, pd.Series(local_aurocs, name=tissue + " (" + str(n_tissue) + ")")], axis = 1)
                df_local_aupr = pd.concat([df_local_aupr, pd.Series(local_auprs, name=tissue + " (" + str(n_tissue) + ")")], axis = 1)
                df_global_auroc = pd.concat([df_global_auroc, pd.Series(global_aurocs, name=tissue + " (" + str(n_tissue) + ")")], axis = 1)
                df_global_aupr = pd.concat([df_global_aupr, pd.Series(global_auprs, name=tissue + " (" + str(n_tissue) + ")")], axis = 1)
        df_local_auroc = df_local_auroc[df_local_auroc.median().sort_values().index[::-1]]
        df_local_aupr = df_local_aupr[df_local_aupr.median().sort_values().index[::-1]]
        df_global_auroc = df_global_auroc[df_global_auroc.median().sort_values().index[::-1]]
        df_global_aupr = df_global_aupr[df_global_aupr.median().sort_values().index[::-1]]
        mean_local_auroc = np.nanmean(df_local_auroc.values)
        mean_local_aupr = np.nanmean(df_local_aupr.values)
        mean_global_auroc = np.nanmean(df_global_auroc.values)
        mean_global_aupr = np.nanmean(df_global_aupr.values)

        plt.figure(figsize=(10,10))

        plt.subplot(2, 2, 1)
        sns.boxplot(data=df_local_auroc, width = 0.9)
        plt.ylabel("local AUROC (mean:" + str(round(mean_local_auroc,4)) + ")")
        plt.xticks(fontsize=8, rotation=90)

        plt.subplot(2, 2, 2)
        sns.boxplot(data=df_local_aupr, width = 0.9)
        plt.ylabel("local AUPR (mean:" + str(round(mean_local_aupr,4)) + ")")
        plt.xticks(fontsize=8, rotation=90)

        plt.subplot(2, 2, 3)
        sns.boxplot(data=df_global_auroc, width = 0.9)
        plt.ylabel("global AUROC (mean:" + str(round(mean_global_auroc,4)) + ")")
        plt.xticks(fontsize=8, rotation=90)

        plt.subplot(2, 2, 4)
        sns.boxplot(data=df_global_aupr, width = 0.9)
        plt.ylabel("global AUPR (mean:" + str(round(mean_global_aupr,4)) + ")")
        plt.xticks(fontsize=8, rotation=90)

        plt.show()

    def show_results_over_epochs(self):
        self.show_classification_results_over_epochs()
        self.show_distortion_results_over_epochs()

    def show_classification_results_over_epochs(self):
        epochs = np.arange(0, self.__config["num_epochs"]+1, self.__config["test_epochs"])

        plt.figure(figsize=(15,10))
        for tissue, model in self.__tissue_specific_embeddings.items():
            results = model.get_evaluation_results()
            if results and ("node_classification" in results):
                for function, result in results["node_classification"].items():
                    plt.subplot(4,1,1)
                    plt.plot(epochs, np.vstack(result["local AUROC"]).mean(axis = 1), label=tissue + "_" + function, alpha = 0.1)
                    plt.subplot(4,1,2)
                    plt.plot(epochs, np.vstack(result["local AUPR"]).mean(axis = 1), label=tissue + "_" + function, alpha = 0.1)
                    plt.subplot(4,1,3)
                    plt.plot(epochs, np.vstack(result["global AUROC"]).mean(axis = 1), label=tissue + "_" + function, alpha = 0.1)
                    plt.subplot(4,1,4)
                    plt.plot(epochs, np.vstack(result["global AUPR"]).mean(axis = 1), label=tissue + "_" + function, alpha = 0.1)
        plt.subplot(4,1,1)
        plt.title("local AUROC over epochs")
        plt.subplot(4,1,2)
        plt.title("local AUPR over epochs")
        plt.subplot(4,1,3)
        plt.title("global AUROC over epochs")
        plt.subplot(4,1,4)
        plt.title("global AUPR over epochs")
        plt.show()

        leaf_local_auroc_means = []
        leaf_local_aupr_means = []
        leaf_local_auroc_stds = []
        leaf_local_aupr_stds = []
        leaf_global_auroc_means = []
        leaf_global_aupr_means = []
        leaf_global_auroc_stds = []
        leaf_global_aupr_stds = []
        for tissue in self.__leaf_tissues:
            results = self.__tissue_specific_embeddings[tissue].get_evaluation_results()
            if results and ("node_classification" in results):
                for function, result in results["node_classification"].items():
                    leaf_local_auroc_means.append(np.vstack(result["local AUROC"]).mean(axis = 1))
                    leaf_local_aupr_means.append(np.vstack(result["local AUPR"]).mean(axis = 1))
                    leaf_local_auroc_stds.append(np.vstack(result["local AUROC"]).std(axis = 1))
                    leaf_local_aupr_stds.append(np.vstack(result["local AUPR"]).std(axis = 1))
                    leaf_global_auroc_means.append(np.vstack(result["global AUROC"]).mean(axis = 1))
                    leaf_global_aupr_means.append(np.vstack(result["global AUPR"]).mean(axis = 1))
                    leaf_global_auroc_stds.append(np.vstack(result["global AUROC"]).std(axis = 1))
                    leaf_global_aupr_stds.append(np.vstack(result["global AUPR"]).std(axis = 1))
        leaf_local_auroc_means = np.vstack(leaf_local_auroc_means).mean(axis=0)
        leaf_local_aupr_means = np.vstack(leaf_local_aupr_means).mean(axis=0)
        leaf_local_auroc_stds = np.vstack(leaf_local_auroc_stds).mean(axis=0)
        leaf_local_aupr_stds = np.vstack(leaf_local_aupr_stds).mean(axis=0)
        leaf_global_auroc_means = np.vstack(leaf_global_auroc_means).mean(axis=0)
        leaf_global_aupr_means = np.vstack(leaf_global_aupr_means).mean(axis=0)
        leaf_global_auroc_stds = np.vstack(leaf_global_auroc_stds).mean(axis=0)
        leaf_global_aupr_stds = np.vstack(leaf_global_aupr_stds).mean(axis=0)

        internal_local_auroc_means = []
        internal_local_aupr_means = []
        internal_local_auroc_stds = []
        internal_local_aupr_stds = []
        internal_global_auroc_means = []
        internal_global_aupr_means = []
        internal_global_auroc_stds = []
        internal_global_aupr_stds = []
        for tissue in self.__internal_tissues:
            results = self.__tissue_specific_embeddings[tissue].get_evaluation_results()
            if results and ("node_classification" in results):
                for function, result in results["node_classification"].items():
                    internal_local_auroc_means.append(np.vstack(result["local AUROC"]).mean(axis = 1))
                    internal_local_aupr_means.append(np.vstack(result["local AUPR"]).mean(axis = 1))
                    internal_local_auroc_stds.append(np.vstack(result["local AUROC"]).std(axis = 1))
                    internal_local_aupr_stds.append(np.vstack(result["local AUPR"]).std(axis = 1))
                    internal_global_auroc_means.append(np.vstack(result["global AUROC"]).mean(axis = 1))
                    internal_global_aupr_means.append(np.vstack(result["global AUPR"]).mean(axis = 1))
                    internal_global_auroc_stds.append(np.vstack(result["global AUROC"]).std(axis = 1))
                    internal_global_aupr_stds.append(np.vstack(result["global AUPR"]).std(axis = 1))
        internal_local_auroc_means = np.vstack(internal_local_auroc_means).mean(axis=0)
        internal_local_aupr_means = np.vstack(internal_local_aupr_means).mean(axis=0)
        internal_local_auroc_stds = np.vstack(internal_local_auroc_stds).mean(axis=0)
        internal_local_aupr_stds = np.vstack(internal_local_aupr_stds).mean(axis=0)
        internal_global_auroc_means = np.vstack(internal_global_auroc_means).mean(axis=0)
        internal_global_aupr_means = np.vstack(internal_global_aupr_means).mean(axis=0)
        internal_global_auroc_stds = np.vstack(internal_global_auroc_stds).mean(axis=0)
        internal_global_aupr_stds = np.vstack(internal_global_aupr_stds).mean(axis=0)

        plt.figure(figsize=(15,10))

        plt.subplot(2,1,1)

        plt.plot(epochs, leaf_local_auroc_means, label="leaf local AUROC")
        plt.fill_between(epochs, y1 = leaf_local_auroc_means - leaf_local_auroc_stds, y2 = leaf_local_auroc_means + leaf_local_auroc_stds, alpha = 0.5, label="leaf local AUROC")
        plt.plot(epochs, leaf_local_aupr_means, label="leaf local AUPR")
        plt.fill_between(epochs, y1 = leaf_local_aupr_means - leaf_local_aupr_stds, y2 = leaf_local_aupr_means + leaf_local_aupr_stds, alpha = 0.5, label="leaf local AUPR")
        
        plt.plot(epochs, internal_local_auroc_means, label="internal local AUROC")
        plt.fill_between(epochs, y1 = internal_local_auroc_means - internal_local_auroc_stds, y2 = internal_local_auroc_means + internal_local_auroc_stds, alpha = 0.5, label="internal local AUROC")
        plt.plot(epochs, internal_local_aupr_means, label="internal local AUPR")
        plt.fill_between(epochs, y1 = internal_local_aupr_means - internal_local_aupr_stds, y2 = internal_local_aupr_means + internal_local_aupr_stds, alpha = 0.5, label="internal local AUPR")

        plt.legend()

        plt.subplot(2,1,2)

        plt.plot(epochs, leaf_global_auroc_means, label="leaf global AUROC")
        plt.fill_between(epochs, y1 = leaf_global_auroc_means - leaf_global_auroc_stds, y2 = leaf_global_auroc_means + leaf_global_auroc_stds, alpha = 0.5, label="leaf global AUROC")
        plt.plot(epochs, leaf_global_aupr_means, label="leaf global AUPR")
        plt.fill_between(epochs, y1 = leaf_global_aupr_means - leaf_global_aupr_stds, y2 = leaf_global_aupr_means + leaf_global_aupr_stds, alpha = 0.5, label="leaf global AUPR")
        
        plt.plot(epochs, internal_global_auroc_means, label="internal global AUROC")
        plt.fill_between(epochs, y1 = internal_global_auroc_means - internal_global_auroc_stds, y2 = internal_global_auroc_means + internal_global_auroc_stds, alpha = 0.5, label="internal global AUROC")
        plt.plot(epochs, internal_global_aupr_means, label="internal global AUPR")
        plt.fill_between(epochs, y1 = internal_global_aupr_means - internal_global_aupr_stds, y2 = internal_global_aupr_means + internal_global_aupr_stds, alpha = 0.5, label="internal global AUPR")

        plt.legend()

        plt.show()

    def show_distortion_results_over_epochs(self):
        epochs = np.arange(0, self.__config["num_epochs"]+1, self.__config["test_epochs"])

        plt.figure(figsize=(15,4))
        for tissue, model in self.__tissue_specific_embeddings.items():
            results = model.get_evaluation_results()
            if results and ("embedding_distortion" in results):
                plt.plot(epochs, results["embedding_distortion"], label=tissue, alpha = 0.1)
        plt.title("Distortion over epochs")
        plt.show()

        leaf_distortions = []
        for tissue in self.__leaf_tissues:
            results = self.__tissue_specific_embeddings[tissue].get_evaluation_results()
            if results and ("embedding_distortion" in results):
                leaf_distortions.append(results["embedding_distortion"])
        leaf_distortions = np.vstack(leaf_distortions)

        internal_distortions = []
        for tissue in self.__internal_tissues:
            results = self.__tissue_specific_embeddings[tissue].get_evaluation_results()
            if results and ("embedding_distortion" in results):
                internal_distortions.append(results["embedding_distortion"])
        internal_distortions = np.vstack(internal_distortions)

        print(leaf_distortions.mean(axis=0)[-1])
        print(internal_distortions.mean(axis=0)[-1])

        plt.figure(figsize=(15,4))
        plt.plot(epochs, leaf_distortions.mean(axis=0), label="leaf distortions")
        plt.fill_between(epochs, y1 = leaf_distortions.mean(axis=0) - leaf_distortions.std(axis=0), y2 = leaf_distortions.mean(axis=0) + leaf_distortions.std(axis=0), alpha = 0.5, label="leaf distortions")
        plt.plot(epochs, internal_distortions.mean(axis=0), label="internal distortions")
        plt.fill_between(epochs, y1 = internal_distortions.mean(axis=0) - internal_distortions.std(axis=0), y2 = internal_distortions.mean(axis=0) + internal_distortions.std(axis=0), alpha = 0.5, label="internal distortions")
        plt.legend()
        plt.show()