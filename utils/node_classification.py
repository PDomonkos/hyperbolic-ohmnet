import torch
from utils.hyperbolic_utils import HypLinear, lorentzian_distance_matrix, poincare_distance

import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.metrics.pairwise import euclidean_distances
    
    

class LinearLayer(torch.nn.Module):  
 
    def __init__(self, input_dim, manifold):
        super(LinearLayer, self).__init__()
        if manifold == None:
            self.__linear = torch.nn.Linear(in_features = input_dim, out_features = 1, bias = True)
        else:
            self.__linear = HypLinear(in_features = input_dim-1, out_features = 1, manifold = manifold)
            
    def forward(self, x):
        return self.__linear(x)



class BCHuberLoss(torch.nn.Module):
    """Code adapted from: https://gist.github.com/thatgeeman/0e48020ecd3d9df65ce77b6956f76062"""
    def __init__(self, pos_weight = None, reduction = "mean"):
        super(BCHuberLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, input, target):
        loss = torch.where(
            target * input > -1,
            (1 - target * input).clamp(min=0).pow(2),
            -4 * target * input,
        )
        if self.pos_weight is not None:
            self.pos_weight = self.pos_weight.to(loss.device)
            loss = torch.where(
                target == 1,
                self.pos_weight * loss,
                loss,
            )
        return loss.mean() if self.reduction == "mean" else loss.sum()



def train_linear_classification(x_train, y_train, x_test, y_test, config):
    classifier = LinearLayer(x_train.shape[1], config["manifold"]).to(config["device"])

    optimizer = torch.optim.SGD(classifier.parameters(), 0.5, momentum=0.0, weight_decay=0.1) 

    criterion = BCHuberLoss(pos_weight = torch.Tensor([y_train.shape[0] / y_train.sum()]))
    y_train[y_train == 0] = -1

    classifier.train()
    x = torch.Tensor(x_train).to(config["device"])
    y = torch.Tensor(np.expand_dims(y_train, axis=1)).to(config["device"])
    for i in range(20):
        optimizer.zero_grad()
        outputs = classifier(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    classifier.eval()
    y_score = classifier(torch.Tensor(x_test).to(config["device"])).detach().cpu().numpy()

    auroc = roc_auc_score(y_test, y_score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    aupr = auc(recall, precision)

    return auroc, aupr



def train_distance_based_classification(x_train, y_train, x_test, y_test, config):
    if config["manifold"] == None:
        train_train_dist = euclidean_distances(x_train)
        test_train_dist = euclidean_distances(x_test, x_train)
    elif str(config["manifold"]).split(" ")[0] == "Lorentz":
        train_train_dist = lorentzian_distance_matrix(torch.Tensor(x_train).detach().cpu(), torch.Tensor(x_train).detach().cpu(), k = config["manifold"].k.detach().cpu()).detach().cpu().numpy()
        test_train_dist = lorentzian_distance_matrix(torch.Tensor(x_test).detach().cpu(), torch.Tensor(x_train).detach().cpu(), k = config["manifold"].k.detach().cpu()).detach().cpu().numpy()
    else:
        x = np.vstack([x_train, x_test])
        dist_mat = poincare_distance(x)
        train_train_dist = dist_mat[:,:x_train.shape[0]][:x_train.shape[0],:]
        test_train_dist = dist_mat[:,:x_train.shape[0]][x_train.shape[0]:,:]

    neigh = KNeighborsClassifier(n_neighbors=10, metric="precomputed")
    neigh.fit(train_train_dist, y_train)
    y_score = neigh.predict_proba(test_train_dist)[:,1]

    auroc = roc_auc_score(y_test, y_score)
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    aupr = auc(recall, precision)

    return auroc, aupr



def train_classification(x_train, y_train, x_test, y_test, config):
    local_auroc, local_aupr = train_distance_based_classification(x_train, y_train, x_test, y_test, config)
    global_auroc, global_aupr = train_linear_classification(x_train, y_train, x_test, y_test, config)

    return local_auroc, local_aupr, global_auroc, global_aupr