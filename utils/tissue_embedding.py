import os
import pickle

import numpy as np

from sklearn.model_selection import StratifiedKFold

import torch
import torch_geometric
import geoopt
from utils.node2manifoldvec import Node2ManifoldVec

from utils.node_classification import train_classification

from utils.hyperbolic_utils import euclidean_distance, poincare_mean, poincare_distance, lorentz_centroid, squared_lorentzian_distance, lorentzian_distance_matrix



class TissueEmbedding():
    """ 
        Abstract superclass for tissue PPI embeddings.
    
        Attributes
        ----------
        tissue_name: str
            Name of the tissue.
        config: dist
            Hyperparameter configuration.
        parent: TissueEmbedding
            Embedding class of the parent tissue in the tissue hierarchy.
        entrez_ids: list
            Entrez IDs of the embedded genes.
        results: dict
            Dictionary storing the evaluation results during training.
        labels: dict
            Dictionary storing GO functions and corresponding labels to the stored genes.
        kfold: sklearn.model_selection.StratifiedKFold
            StratifiedKFold class for evaluating the node classification task.
        shortest_path_distance_matrix: np.array
            Shortest path distances in the largest connected component of the PPI network.
        shortest_path_distance_entrez_ids: np.array
            Entrez IDs corresponding to the largest connected component of the PPI network.

        Methods
        -------
        init(tissue_name, config)
            Initialize the model.
        set_parent(parent_tissue)
            Store the parent tissue given in the parameter.
        set_node_labels(labels, n_splits)
            Initialize a StratifiedKFold class based on the number of splits given as a parameter.
            Initialize the result dictionary for each GO function in the given labels dictionary.
            Store the node labels for functions where there are at least n_splits positive class labels.
        get_node_labels(): dict
            Return the stored node labels for each GO function. 
        set_graph_distances(distance_matrix, entrez_ids)
            Initialize the results and store the given shortest path distances and entrez IDs corresponding to the largest connected component.
        get_entrez_ids()
            Return the Entrez IDs corresponding to the embedded genes.
        evaluate_node_classification(save): dict
            Perform a cross-validation for each GO function in the stored labels based on the StratifiedKFold split.
            Return the resulting ROCAUC and PRAUC scores, store the results if the "save" parameter is true.
            Each AUC score has a local and a global version corresponding to different classifiers:
                local refers to predicting labels based on local neighbourhoods with a KNN classifier,
                global refers to predicting labels based on one decision surface with a logistic regression model.
        evaluate_graph_distortion(save): float
            Calculate the distortion between shortest path distances and embedding distances on the largest connected component.
            Return the resulting distortion score, store the result if the "save" parameter is true.
        evaluate_embeddings(save, classification, distortion): dict
            Evaluate node classification (if the "classification" parameter is True) and/or embedding distortion (if the "distortion" parameter is True).
            Return the resulting scores in a dict, store the results if the "save" parameter is true.
        get_evaluation_results()
            Return the stored evaluation results.
        save(path)
            Save the stored results to the given path.
        load(path)
            Load and store previous results from the given path.
        average_embeddings(embeddings): torch.Tensor
            Calculate the average of the given embeddings.
        get_average_embedding()
            Return the average of all stored embeddings.
        get_embeddings(): torch.Tensor
            Abstract method: return the stored embeddings.
        update_embeddings()
            Abstract method: update the stored embeddings.
    """

    def __init__(self, tissue_name, config):
        self._tissue_name = tissue_name
        self._config = config
        self._parent = None
        self._labels = None
        self._shortest_path_distance_matrix = None
        self._results = None
        self._entrez_ids = []

    def set_parent(self, parent_tissue):
        self._parent = parent_tissue

    def set_node_labels(self, labels, n_splits = 5):
        if self._results is None:
            self._results = dict()
        self._results["node_classification"] = dict()
        self._labels = dict()
        self._kfold = StratifiedKFold(n_splits = n_splits, random_state = 42, shuffle = True)
        for function, label in labels.items():
            l = np.array([label[gene] for gene in self._entrez_ids]).astype(int)
            if l.sum() >= n_splits:
                self._results["node_classification"][function] = {"local AUROC": [], "local AUPR": [], "global AUROC": [], "global AUPR": []}
                self._labels[function] = l

    def get_node_labels(self):
        return self._labels

    def set_graph_distances(self, distance_matrix, entrez_ids):
        if self._results is None:
            self._results = dict()
        self._results["embedding_distortion"] = []
        self._shortest_path_distance_matrix = distance_matrix
        self._shortest_path_distance_entrez_ids = entrez_ids

    def get_entrez_ids(self):
        return self._entrez_ids

    def evaluate_node_classification(self, save):
        res = None
        if self._labels is not None:
            x = self.get_embeddings()
            x = x.detach().cpu().numpy()
            res = dict()
            for function, y in self._labels.items():
                res[function] = {"local AUROC": np.zeros((self._kfold.get_n_splits(),)), 
                                 "local AUPR": np.zeros((self._kfold.get_n_splits(),)),
                                 "global AUROC": np.zeros((self._kfold.get_n_splits(),)), 
                                 "global AUPR": np.zeros((self._kfold.get_n_splits(),))}
                for i, (train_index, test_index) in enumerate(self._kfold.split(x, y)):
                    local_auroc, local_aupr, global_auroc, global_aupr = train_classification(x[train_index], y[train_index], x[test_index], y[test_index], self._config)
                    res[function]["local AUROC"][i] = local_auroc
                    res[function]["local AUPR"][i] = local_aupr
                    res[function]["global AUROC"][i] = global_auroc
                    res[function]["global AUPR"][i] = global_aupr
                if save:
                    self._results["node_classification"][function]["local AUROC"] += [res[function]["local AUROC"]]
                    self._results["node_classification"][function]["local AUPR"] += [res[function]["local AUPR"]]
                    self._results["node_classification"][function]["global AUROC"] += [res[function]["global AUROC"]]
                    self._results["node_classification"][function]["global AUPR"] += [res[function]["global AUPR"]]
        return res

    def evaluate_graph_distortion(self, save):
        """
        Learning the best constant C is adapted from:
            https://github.com/mcneela/Mixed-Curvature-Pathways/blob/master/multiset/min_distortion_multiple.py
            Apache License, Version 2.0
            Copyright 2023 Daniel McNeela, Albert Gu, Frederic Sala, Beliz Gunel, Christopher Ré
        """
        distortion = None
        if self._shortest_path_distance_matrix is not None:
            triu_indices = np.triu_indices(self._shortest_path_distance_matrix.shape[0], k = 1)
            graph_distances = torch.Tensor(self._shortest_path_distance_matrix[triu_indices]).detach()

            model_indices = [self.get_entrez_ids().index(i) for i in self._shortest_path_distance_entrez_ids]
            embeddings = self.get_embeddings().cpu()[model_indices]

            if self._config["manifold"] == None:
                embedding_distances = euclidean_distance(embeddings)
            elif str(self._config["manifold"]).split(" ")[0] == "Lorentz": 
                embedding_distances = lorentzian_distance_matrix(embeddings, embeddings, self._config["manifold"].k.detach().cpu())
            else:
                embedding_distances = torch.Tensor(poincare_distance(embeddings.numpy())).detach()
            embedding_distances = embedding_distances[triu_indices]

            C = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32))
            optimizer = torch.optim.Adam([C], lr=0.5)
            for i in range(100):
                optimizer.zero_grad()
                distortion = torch.mean(torch.abs(C * embedding_distances / graph_distances - 1))
                distortion.backward()
                optimizer.step()
            distortion = distortion.item()

            if save:
                self._results["embedding_distortion"] += [distortion]

        return distortion
    
    def evaluate_embeddings(self, save, classification = True, distortion = True):
        result = dict()
        if classification:
            result["node_classification"] = self.evaluate_node_classification(save)
        if distortion:
            result["embedding_distortion"] = self.evaluate_graph_distortion(save)
        return result
    
    def get_evaluation_results(self):
        return self._results

    def save(self, path):
        if self._results is not None:
            with open(os.path.join(path, self._tissue_name + "_results.pickle"), 'wb') as f:
                pickle.dump(self._results, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        if self._results is not None:
            with open(os.path.join(path, self._tissue_name + "_results.pickle"), 'rb') as f:
                self._results = pickle.load(f)
    
    def _average_embeddings(self, embeddings):
        if self._config["manifold"] == None:
            return embeddings.mean(dim=0)
        elif str(self._config["manifold"]).split(" ")[0] == "PoincareBallExact(exact)":
            return poincare_mean(embeddings, dim=0, k=-self._config["manifold"].k)
        elif str(self._config["manifold"]).split(" ")[0] == "Lorentz":
            return lorentz_centroid(embeddings, self._config["manifold"], w=None)
        
    def get_average_embedding(self):
        return self._average_embeddings(self.get_embeddings())

    def get_embeddings(self):
        raise NotImplementedError
    
    def update_embeddings(self):
        raise NotImplementedError


    
class LeafTissueEmbedding(TissueEmbedding):
    """
        Tissue embedding class for PPI networks in the leaf nodes of the tissue hierarchy tree.

        Attributes
        ----------
        model: torch.nn.Module
            Node2Vec model for embedding the PPI network.
        optimizer: torch.optim.Optimizer
            Optimizer for the weights of the Node2Vec model.
            
        Methods
        ----------
        init(tissue_name, config, ppi_graph)
            Store the Entrez IDs, initialize the Node2Vec model and optimizer.
        save(path)
            Save the stored results and the model parameters to the given path.
        load(path)
            Load and store previous results and the model parameters from the given path.
        get_embeddings(): torch.Tensor
            Return the Node2Vec embeddings.
        update_embeddings()
            Train the Node2Vec model for one epoch.
        parent_regularization(regularization_indices): torch.Tensor
            Return the regularization term based on the distances of the gene embeddings in the leaf node and the corresponding gene embeddings in the parent tissue.
    """

    def __init__(self, tissue_name, config, ppi_graph):
        super(LeafTissueEmbedding, self).__init__(tissue_name, config)

        self._entrez_ids = [n.split("__")[1] for n in np.array(list(ppi_graph.nodes))]

        if self._config["manifold"] == None:
            self.__model = torch_geometric.nn.Node2Vec(torch_geometric.utils.convert.from_networkx(ppi_graph).edge_index, self._config["embedding_dim"], 
                                                   self._config["walk_length"], self._config["context_size"], self._config["walks_per_node"], 
                                                   self._config["p"], self._config["q"], self._config["num_negative_samples"], 
                                                   ppi_graph.number_of_nodes(), sparse=False).to(self._config["device"])
            self.__optimizer = torch.optim.Adam(list(self.__model.parameters()), lr=self._config["learning_rate"], betas=(0.0, 0.99))
        else:
            if str(config["manifold"]).split(" ")[0] == "Lorentz":
                ambient_dim = 1
            else:
                ambient_dim = 0
            self.__model = Node2ManifoldVec(self._config["manifold"], torch_geometric.utils.convert.from_networkx(ppi_graph).edge_index, self._config["embedding_dim"] + ambient_dim, 
                                                   self._config["walk_length"], self._config["context_size"], self._config["walks_per_node"], 
                                                   self._config["p"], self._config["q"], self._config["num_negative_samples"], 
                                                   ppi_graph.number_of_nodes()).to(self._config["device"])
            self.__optimizer = geoopt.optim.RiemannianAdam(params=list(self.__model.parameters()), lr=self._config["learning_rate"], betas=(0.0, 0.99), stabilize=1)

        self.__loader = self.__model.loader(batch_size=self._config["batch_size"], shuffle=True, num_workers=self._config["num_workers"])

    def save(self, path):
        super().save(path)
        torch.save(self.__model.state_dict(), os.path.join(path, self._tissue_name + "_embeddings.pt"))

    def load(self, path):
        super().load(path)
        self.__model.load_state_dict(torch.load(os.path.join(path, self._tissue_name + "_embeddings.pt")))
        self.__model.eval()
    
    @torch.no_grad()
    def get_embeddings(self):
        self.__model.eval()
        return self.__model().detach()

    def update_embeddings(self):
        self.__model.train()
        for pos_rw, neg_rw in self.__loader:
            self.__optimizer.zero_grad()
            node2vec_loss = self.__model.loss(pos_rw.to(self._config["device"]), neg_rw.to(self._config["device"]))
            regularization_loss = self.__parent_regularization(pos_rw[:, 0])
            loss = node2vec_loss + self._config["lambda"] * regularization_loss
            loss.backward()
            self.__optimizer.step()

    def __parent_regularization(self, regularization_indices):
        regularization_embeddings = self.__model()[regularization_indices]
        regularization_parent_indices = [self._parent.get_entrez_ids().index(self._entrez_ids[i]) for i in regularization_indices]
        regularization_parent_embeddings = self._parent.get_embeddings()[regularization_parent_indices]

        if self._config["manifold"] == None:
            regularization = 1/2 * torch.sum(torch.pow(regularization_embeddings - regularization_parent_embeddings, 2), dim = 1).mean()
        elif str(self._config["manifold"]).split(" ")[0] == "PoincareBallExact(exact)":
            regularization = self._config["manifold"].dist(regularization_embeddings, regularization_parent_embeddings).mean()
        elif str(self._config["manifold"]).split(" ")[0] == "Lorentz":
            regularization = squared_lorentzian_distance(regularization_embeddings, regularization_parent_embeddings, manifold = self._config["manifold"]).mean()

        return regularization



class InternalTissueEmbedding(TissueEmbedding):
    """
        Tissue embedding class for PPI networks in the internal nodes of the tissue hierarchy tree.

        Attributes
        ----------
        children: list
            List of the EmbeddingNodes corresponding to the child tissues in the hierarchy tree.

        Methods
        ----------
        init(tissue_name, config, children)
            Set the child-parent relationships and store the union of the children's Entrez IDs.
            Initialize embeddings as the average of the parent and children's corresponding gene embeddings.
        save(path)
            Save the stored results and the embeddings to the given path.
        load(path)
            Load and store previous results and the embeddings from the given path.
        get_embeddings(): torch.Tensor
            Return the stored embeddings.
        update_embeddings()
            Set the gene embeddings to the average embeddings of the corresponding genes in the parent and children nodes.
    """

    def __init__(self, tissue_name, config, children):
        super(InternalTissueEmbedding, self).__init__(tissue_name, config)

        self.__children = children
        for child in self.__children:
            child.set_parent(self)
            for gene in child.get_entrez_ids():
                if gene not in self._entrez_ids:
                    self._entrez_ids.append(gene)   
        
        self.update_embeddings()

    def save(self, path):
        super().save(path)
        torch.save(self.__embeddings, os.path.join(path, self._tissue_name + "_embeddings.pt"))

    def load(self, path):
        super().load(path)
        self.__embeddings = torch.load(os.path.join(path, self._tissue_name + "_embeddings.pt"))
    
    def get_embeddings(self):
        return self.__embeddings

    def update_embeddings(self):
        collected_embeddings = dict(zip(self._entrez_ids, [[]] * len(self._entrez_ids)))
        for child in self.__children:
            for gene, embedding in zip(child.get_entrez_ids(), child.get_embeddings()):
                collected_embeddings[gene] = collected_embeddings[gene] + [embedding]
        if self._parent:
            for gene, embedding in zip(self._parent.get_entrez_ids(), self._parent.get_embeddings()):
                if gene in collected_embeddings:
                    collected_embeddings[gene].append(embedding)

        self.__embeddings = []
        for gene in self._entrez_ids:
            self.__embeddings.append(self._average_embeddings(torch.stack(collected_embeddings[gene], dim=0)))

        self.__embeddings = torch.stack(self.__embeddings, dim=0).detach()