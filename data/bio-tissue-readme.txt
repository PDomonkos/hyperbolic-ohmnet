Multi-layer human tissue network

# About

edgelists:
	Network layers. Each file contains edge information for one network layer representing human protein-protein interaction (PPI) network in one tissue (e.g., tooth.edgelist is tooth-specific PPI network). Nodes are human genes specifically active in that tissue. Node IDs are Entrez gene IDs (integers, but not consecutive). The same node ID appearing in two edgelists represents the same gene. 

hierarchy: 
	Tissue hierarchy. The same information is given in multiple different formats (e.g., edge list, OBO ontology, root-node paths).

labels:
	Node labels. E.g., smooth_muscle_GO:0048661.lab contains label information for nodes in network layer smooth_muscle. Each row in smooth_muscle_GO:0048661.lab has the label for one node (gene): 1 - gene has a function GO:0048661 in smooth muscle tissue, 0 - gene does not (i.e., is not known to) have a function GO:0048661 in smooth muscle tissue. Consult Gene Ontology (http://geneontology.org) for information about gene functions. E.g., GO:0048661 is "positive regulation of smooth muscle cell proliferation".

# Contact

Data was collected by Marinka Zitnik in August-September 2016. Preprocessing was done according to the established practice in computational biology.

marinka@cs.stanford.edu

# Cite

Predicting multicellular function through multi-layer tissue networks.
Marinka Zitnik and Jure Leskovec.
Bioinformatics, 2017.
Presented at ISMB/ECCB 2017.
