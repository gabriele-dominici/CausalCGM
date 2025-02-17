# These functions are modified functions starting from here https://github.com/kevinsbello/dagma/
import torch
import numpy as np 
import torch
import torch.nn as nn
import numpy as np
import typing
import torch.nn.functional as F

class DagmaCE(nn.Module):
    """
    Class that models the structural equations for the causal graph using MLPs.
    """

    def __init__(self, n_concepts: int, n_classes: int, dims: typing.List[int], bias: bool = False, gamma: float = 10.0):
        r"""
        Parameters
        ----------
        n_concepts : typing.List[int]
            Number of concepts in the dataset.
        n_classes : typing.List[int]
            Number of classes in the dataset.
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        """
        super(DagmaCE, self).__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_symbols = n_concepts + n_classes
        self.dims = dims
        self.I = torch.eye(self.n_symbols)
        self.mask = torch.ones(self.n_symbols, self.n_symbols)
        # remove self loop 
        self.mask = self.mask - self.I
        # remove edges from classes to concepts
        self.mask[n_concepts:] = torch.zeros(n_classes, self.n_symbols)

        self.edges_to_check = []
        self.edge_matrix = torch.nn.Parameter(torch.zeros(self.n_symbols, self.n_symbols))
        self.family_dict = None

        self.family_of_concepts = None

        # threshold for the adjacency matrix as hyperparameter
        self.th = torch.tensor([0.02])
        self.gamma = gamma
        self.fc1 = nn.Linear(self.n_symbols, self.n_symbols, bias=bias)
        layers = []
        for lid, l in enumerate(range(len(dims) - 1)):
            layers.append(nn.Linear(dims[l], dims[l + 1], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    # def edge_to_check(self, edge, family_of_concepts=None):
    #     self.add_edges = []
    #     self.edges_to_check = edge
    #     for _ in edge:
    #         self.add_edges += [nn.Parameter(torch.tensor([0.0]))]
        # if family_of_concepts is not None:
        #     self.set_family_of_concepts(family_of_concepts)

    # def set_family_of_concepts(self, family_of_concepts):
    #     self.family_of_concepts = family_of_concepts
    #     self.family_dict = {}
    #     count = 0
    #     for i, el in enumerate(family_of_concepts):
    #         self.family_dict[i] = list(range(count, count + el))
    #         count += el

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        fc1_weight = self.fc1_to_adj()
        A = torch.abs(fc1_weight)  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.n_symbols * np.log(s)
        return torch.abs(h)
    
    def edge_loss(self):
        """
        Loss function that is minimized, increasing the number of edges in the graph.
        Not used!
        Returns
        -------
        torch.Tensor
            A scalar value of the edge loss.
        """
        fc1_weight = self.fc1_to_adj()
        n_edges = fc1_weight[:self.n_concepts, :self.n_concepts].sum()
        return 1/(n_edges + 1)#  + (n_y-1)**2  



    def fc1_to_adj(self, train=False) -> torch.Tensor:  # [j * m1, i] -> [i, j]
        r"""
        Computes the induced weighted adjacency matrix W from the first FC weights.
        Intuitively each edge weight :math:`(i,j)` is the *L2 norm of the functional influence of variable i to variable j*.

        Returns
        -------
        np.ndarray
            :math:`(d,d)` weighted adjacency matrix
        """
        # put to 0 the diagonal and the edges from classes to concepts
        W = torch.abs(self.fc1.weight * self.mask)
        # use the edge matrix (learnable parameters) to discover the edges to check (either A->B or B->A)
        # iterating through the edges to check, to see if A->B is 1 or 0
        for i, el in enumerate(self.edges_to_check):
        #     if self.family_of_concepts is not None:
        #         index1 = self.family_dict[el[0]][0]
        #         index2 = self.family_dict[el[1]][0]
        #         mask_tmp[index1, index2] += 1
        #     else:
            W[el[0], el[1]] += torch.sigmoid(self.edge_matrix[el[0], el[1]])
        # iterating through the edges to check, to put the opposite value of what we found in the previous step
        for i, el in enumerate(self.edges_to_check):
        #     if self.family_of_concepts is not None:
        #         index1 = self.family_dict[el[1]][0]
        #         index2 = self.family_dict[el[0]][0]
        #         W[index1, index2] += 1 - (W[index2, index1] > 0.5).float()
        #     else:
            W[el[1], el[0]] += 1 - W[el[0], el[1]]
        # normalize and threshold to remove the weakest edges

        W_mask = torch.sigmoid(5 * (W - self.th))
        # Straight-Through Estimator: Use binary mask in forward pass, but sigmoid in backward pass
        W_mask = W_mask + ((W > self.th).float() - W_mask).detach()
        # W_mask = W > self.th
        # W = torch.abs(self.combined_sigmoid(W, k1=3, x1=-2, k2=3, x2=2))
        W = W * W_mask
        
        # mask_representer = torch.ones_like(W)
        # if self.family_of_concepts is not None:
        #     for _, value in self.family_dict.items():
        #         idxs = value[1:]
        #         for idx in idxs:
        #             mask_representer[idx, :] = 0
        #             mask_representer[:, idx] = 0
        #     W = W * mask_representer
        #     if self.family_of_concepts is not None:
        #         for _, value in self.family_dict.items():
        #             idx_to_copy = value[0]
        #             for el in value[1:]:
        #                 W[:, el] += W[:, idx_to_copy]
        #                 W[el, :] += W[idx_to_copy, :]
        return W 

class CausalLayer(DagmaCE):
    """
    Class that models the structural equations for the causal graph using MLPs.
    """

    def __init__(self, n_concepts: int, n_classes: int, dims: typing.List[int], bias: bool = False, gamma: float = 10.0):
        r"""
        Parameters
        ----------
        n_concepts : typing.List[int]
            Number of concepts in the dataset.
        n_classes : typing.List[int]
            Number of classes in the dataset.
        dims : typing.List[int]
            Number of neurons in hidden layers of each MLP representing each structural equation.
        bias : bool, optional
            Flag whether to consider bias or not, by default ``True``
        """
        super().__init__(n_concepts, n_classes, dims, bias, gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n, d] -> [n, d]
        r"""
        Applies the current states of the structural equations to the dataset X

        Parameters
        ----------
        x : torch.Tensor
            Input dataset with shape :math:`(n,d)`.

        Returns
        -------
        torch.Tensor
            Result of applying the structural equations to the input data.
            Shape :math:`(n,d)`.
        """
        fc1_weight = self.fc1_to_adj()
        x = torch.matmul(x, fc1_weight)
        x = x.permute(0, 2, 1)
        for fc in self.fc2:
            x = F.leaky_relu(x)
            x = fc(x)
        x = F.leaky_relu(x)
        return x
  
