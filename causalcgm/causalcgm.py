import torch
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import torch
import networkx as nx
import random
import numpy as np 
from causalcgm.utils import cace_score
from sympy.logic import SOPform
from sympy import symbols, true, false
import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm.auto import tqdm
import typing
import torch.nn.functional as F
from causalcgm.dagma import CausalLayer


class CausalConceptGraphLayer(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            n_classes,
            emb_size,
            gamma=10.0,
    ):
        """
        Parameters
        ----------
        in_features : int
            Number of input features.
        n_concepts : int
            Number of concepts.
        n_classes : int
            Number of classes.
        emb_size : int
            Embedding size.
        gamma : float
            DAGMA parameter.
        """
        super().__init__()
        self.in_features = in_features
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.n_symbols = n_concepts + n_classes

        self.eq_model = CausalLayer(self.n_concepts, self.n_classes,
                                    [self.emb_size, self.emb_size*2], bias=False,
                                    gamma=gamma)
        
        self.concept_context_generators = torch.nn.ModuleList()
        self.concept_prob_predictor = torch.nn.ModuleList()
        self.concept_prob_predictor_post = torch.nn.ModuleList()
        for i in range(self.n_symbols):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
            self.concept_prob_predictor.append(torch.nn.Sequential(
                torch.nn.Linear(2 * emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, 1)
            ))
            self.concept_prob_predictor_post.append(torch.nn.Sequential(
                torch.nn.Linear(2 * emb_size, emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(emb_size, 1)
            ))
        self.reshaper = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )

        self.parent_indices = None
        self.scm = None

    def _build_concept_embedding(self, context_mix, c_pred):
        context_pos = context_mix[:, :self.emb_size]
        context_neg = context_mix[:, self.emb_size:]
        return context_pos * c_pred + context_neg * (1 - c_pred)

    def edges_in_cycles(self, graph):
        """
        Find all edges that are contained in at least one cycle in a directed graph.

        :param graph: A NetworkX directed graph (nx.DiGraph)
        :return: A set of edges that are part of some cycle
        """
        cycle_edges = set()
        
        # Compute Strongly Connected Components (SCCs)
        sccs = list(nx.strongly_connected_components(graph))
        
        for scc in sccs:
            if len(scc) > 1:  # Only consider SCCs that contain cycles
                # Extract edges inside this SCC
                for node in scc:
                    for neighbor in graph.successors(node):
                        if neighbor in scc:
                            cycle_edges.add((node, neighbor))
        
        return list(cycle_edges)

    def compute_parent_indices(self):
        """
        Compute parent indices from the adjacency matrix. (self.parent_indices)
        And makes scm a DAG if it is not cutting off least important edges.
        """
        self.scm = torch.FloatTensor(self.eq_model.fc1_to_adj()) 
        scm = self.scm.detach().numpy()
        check = False
        while check == False:
            try:
                G = nx.from_numpy_array(scm, create_using=nx.DiGraph)
                self.parent_indices = {node: list(G.predecessors(node)) for node in list(nx.topological_sort(G))}
                check = True
            except:
                scm_tmp = scm.copy()
                G = nx.from_numpy_array(scm, create_using=nx.DiGraph)
                # if self.eq_model.family_dict is not None:
                #     rows = []
                #     for key, value in self.eq_model.family_dict.items():
                #         rows += [np.expand_dims(scm[value[0], :], axis=0)]
                #     rows = np.concatenate(rows, axis=0)
                #     cols = []
                #     for key, value in self.eq_model.family_dict.items():
                #         cols += [np.expand_dims(rows[:, value[0]], axis=1)]
                #     scm_tmp = np.concatenate(cols, axis=1)
                # find all the connections that are inside some cycles
                cycles_edges = self.edges_in_cycles(G)
                scm_tmp[scm_tmp == 0] = 100
                for i in range(scm_tmp.shape[0]):
                    for j in range(scm_tmp.shape[1]):
                        if (i, j) not in cycles_edges:
                            scm_tmp[i, j] = 100
                index = np.unravel_index(np.argmin(scm_tmp), scm_tmp.shape)
                # if self.eq_model.family_dict is not None:
                #     for el in self.eq_model.family_dict[index[0]]:
                #         for el2 in self.eq_model.family_dict[index[1]]:
                #             scm[el, el2] = 0
                #     # print(f'Remove edge: {index[0]} -> {index[1]}')
                # else:
                    # print(f'Remove edge: {index}')
                scm[index] = 0
                self.dag = scm.copy()
        scm_tmp = scm.copy()
        # if self.eq_model.family_dict is not None:
        #     rows = []
        #     for key, value in self.eq_model.family_dict.items():
        #         rows += [np.expand_dims(scm[value[0], :], axis=0)]
        #     rows = np.concatenate(rows, axis=0)
        #     cols = []
        #     for key, value in self.eq_model.family_dict.items():
        #         cols += [np.expand_dims(rows[:, value[0]], axis=1)]
        #     scm_tmp = np.concatenate(cols, axis=1)
        self.dag = scm_tmp.copy()
        

    def forward(self, x, intervention_idxs=[], c=None, train=False):
        """
        Forward pass of the CausalConceptGraphLayer.
        If train is True, we predict all the concepts c from x and all the concepts copies c' from ground truth c that are 
        connected to in the matrix self.dag.
        If train is False (inference), we predict the root of the DAG from x (as they would be c) and then we traverse the DAG to predict the rest
        of the concepts, as they would be c'.
        """
        if train:
            # if self.eq_model.family_dict is not None:
            #     int_list = list(range(len(self.eq_model.family_dict.keys())))
            #     n = 1
            #     selected_key = random.sample(int_list, n)[0]
            #     selected_elements = torch.tensor(self.eq_model.family_dict[selected_key])
            #     I = torch.eye(len(self.eq_model.family_dict[selected_key]))
            #     # random index to select a row in I 
            #     idx1 = torch.randint(0, I.shape[0], (x.shape[0],))
            #     # another random index to select a row in I except idx1
            #     idx2 = torch.randint(0, I.shape[0], (x.shape[0],))
            #     filter_wrong_idx = idx2 == idx1
            #     idx2[filter_wrong_idx] = (idx2[filter_wrong_idx] + 1) % I.shape[0]
            #     c0 = (torch.randn((1, self.n_symbols)) > 0.5).float().repeat(x.shape[0], 1)
            #     c0[:, selected_elements] = I[idx1]
            #     c1 = (torch.randn((1, self.n_symbols)) > 0.5).float().repeat(x.shape[0], 1)
            #     c1[:, selected_elements] = I[idx2]
            # else:

            # train forward pass, we want to intervene on all the concepts values to predict c' to reduce leakage
            int_idx = torch.arange(self.n_concepts) # intervene on all the concepts values
            s_preds_prior, s_preds_posterior, s_emb_prior, s_emb_posterior, context_posterior = self._train_forward(x, c=c)

            # To take into account the average treatment effect of interventions in the loss function, to make the model more responsive to interventions
            # we need to intervene on a random subset of the concepts values

            int_list = list(range(self.n_concepts))  

            # The number of elements you want to select at the same time
            n = 1 

            # Selecting n random elements from the list
            selected_elements = [torch.tensor(random.sample(int_list, n)) for i in range(x.shape[0])]

            # intervene on the selected concepts values
            c0 = c.clone()# (torch.randn((1, self.n_symbols)) > 0.5).float().repeat(x.shape[0], 1)
            c1 = c.clone()
            for i, selected_elements in enumerate(selected_elements):
                c0[i, selected_elements] = 0
                c1[i, selected_elements] = 1

            # forward pass with intervened concepts values fixed to 0 and 1 to get the average treatment effect
            s_preds_prior0, s_preds_posterior0, _, _, _ = self._train_forward(x, c0)
            s_preds_prior1, s_preds_posterior1, _, _, _ = self._train_forward(x, c1)
            
            # split in concepts and classes
            c_preds_prior = s_preds_prior[:, :self.n_concepts]
            y_preds_prior = s_preds_prior[:, self.n_concepts:]
            c_preds_posterior = s_preds_posterior[:, :self.n_concepts]
            y_preds_posterior = s_preds_posterior[:, self.n_concepts:]
            
            return c_preds_prior, y_preds_prior, c_preds_posterior, y_preds_posterior, s_preds_posterior0, s_preds_posterior1, s_emb_prior, s_emb_posterior, context_posterior

        else:
            # after training we need to compute the DAG and use it for inference
            # all test-time operations will be done starting from the root nodes of the DAG moving forward to the leaves
            self.compute_parent_indices() # compute the DAG and the parent indices to define root nodes and how to traverse the DAG
            is_root = self.scm.sum(dim=0) == 0 # check if a node is a root node
            assert self.parent_indices is not None, "Parent indices not found. Please train DAGMA MLP first."
            assert sum(is_root) > 0, "No root nodes found. Please train DAGMA MLP first or your DAG is still a DAG."

            # forward pass of the CausalConceptGraphLayer during inference
            s_preds_prior, s_preds_posterior, s_emb_prior, s_emb_posterior, _ = self._inference_forward(x, intervention_idxs=intervention_idxs, c=c, return_intervened=True)

            # split in concepts and classes
            c_preds = s_preds_posterior[:, :self.n_concepts]
            y_preds = s_preds_posterior[:, self.n_concepts:]

            return c_preds, y_preds

    def _train_forward(self, x, c):
        """
        Forward pass of the CausalConceptGraphLayer during training.
        """
        c_context_dict = {}
        c_emb_true_dict = [0 for el in range(self.n_symbols)]
        s_prior = [0 for el in range(self.n_symbols)]
        context_posterior_list = []
        # first pass: compute the context (exogenous variable) of each concept from the input embedding and then compute the concept values and embeddings
        for current_node in range(self.n_symbols):
            # compute the context (exogenous variable) of each concept from the input embedding
            context = self.concept_context_generators[current_node](x) # context or an emebdding for each concepts that contains both the positive and negative context see CEM paper for more details
            c_emb_true = self._build_concept_embedding(context, c[:, current_node].unsqueeze(1)) # build the concept embedding using the ground truth concept values - useful to predict c' in the following step
            c_emb_true_dict[current_node] = c_emb_true
            c_context_dict[current_node] = context
            s_prior[current_node] = self.concept_prob_predictor[current_node](c_context_dict[current_node]) # predict the concept value from the context 
        s_prior = torch.cat(s_prior, axis=1)
        # if self.eq_model.family_dict is not None:
        #     s_prior[:, :self.n_concepts] = torch.sigmoid(s_prior[:, :self.n_concepts])
        #     s_prior[:, self.n_concepts:] = torch.softmax(s_prior[:, self.n_concepts:], dim=1)
        # else: 
        s_prior = torch.sigmoid(s_prior)
        s_emb_prior = torch.stack(c_emb_true_dict).permute(1, 0, 2)
        s_pred_list = [0 for el in range(self.n_symbols)]
        c_emb_post_dict = [0 for el in range(self.n_symbols)]
        context_posterior_dict = [0 for el in range(self.n_symbols)]
        # second pass: predict the concept values of the copies from the concept embeddings of its parents according to the DAG
        for current_node in range(self.n_symbols):
            context_posterior = self.eq_model(s_emb_prior.permute(0, 2, 1))[:, current_node] # aggregate the concept embeddings of the parents to get the concept embedding of the parents
            context_posterior_dict[current_node] = context_posterior
            s_pred = self.concept_prob_predictor[current_node](context_posterior) # use the concept embedding of the parents to predict the concept value of the current node
            s_pred_list[current_node] = s_pred
        s_pred_list = torch.cat(s_pred_list, axis=1)
        # if self.eq_model.family_dict is not None:
        #     for key, value in self.eq_model.family_dict.items():
        #         s_pred_list[:, :self.n_concepts] = torch.sigmoid(s_pred_list[:, :self.n_concepts])
        #         s_pred_list[:, self.n_concepts:] = torch.softmax(s_pred_list[:, self.n_concepts:], dim=1)
        # else:
        s_pred_list = torch.sigmoid(s_pred_list)
        # third pass: build the concept embeddings of the copies using the concept values just predicted (from parents) and the context of the current node predicted in the first pass
        for current_node in range(self.n_symbols):
            c_emb_true = self._build_concept_embedding(context_posterior_dict[current_node], s_pred_list[:, current_node].unsqueeze(1))
            c_emb_post_dict[current_node] = c_emb_true
        s_emb_post = torch.stack(c_emb_post_dict).permute(1, 0, 2)
        s_preds = s_pred_list
        
        return s_prior, s_preds, s_emb_prior, s_emb_post, context_posterior_list
    
    def _inference_forward(self, x, intervention_idxs, c, return_intervened=True, sample=False):
        """
        Forward pass of the CausalConceptGraphLayer during inference.
        """
        c_emb_dict = []
        c_context_dict = {}
        s_prior = [0 for el in range(self.n_symbols)]
        
        # compute the DAG and the parent indices to define root nodes and how to traverse the DAG
        self.compute_parent_indices()
        is_root = self.scm.sum(dim=0) == 0 # define root nodes
        s_pred_list = [0 for el in range(self.n_symbols)]
        context_posterior_list = []
        s_preds_tmp = []
        context_posterior_list = []
        concept_group = []
        # traverse the DAG according to the parent indices ordered by topological sort
        for current_node, parent_nodes in self.parent_indices.items():
            # compute the context (exogenous variable) of the current node
            context = self.concept_context_generators[current_node](x)
            c_context_dict[current_node] = context
            # if root node: predict the concept value from the context
            if is_root[current_node]:
                s_pred = self.concept_prob_predictor[current_node](context)
            # else: predict the concept value from the concept embeddings of the parents
            else:
                context_posterior = self.eq_model(c_emb_dict.permute(0, 2, 1))[:, current_node] # aggregate the concept embeddings of the parents to get the concept embedding of the parents
                s_pred = self.concept_prob_predictor[current_node](context_posterior) # use the concept embedding of the parents to predict the concept value of the current node
            # if self.eq_model.family_dict is None or current_node < self.n_concepts:
            s_pred = torch.sigmoid(s_pred)
            # Time to check for interventions, do that if intervention applies to the current concept
            c_int = self._after_interventions(
                prob=s_pred,
                concept_idx=current_node,
                intervention_idxs=intervention_idxs,
                c_true=c,
            )
            # if intervention applies to current concept: use ground truth values
            if current_node in intervention_idxs:
                c_to_use = c_int
                s_pred_list[current_node] = (c_int)
            # else: use predictions
            else:
                c_to_use = s_pred
                s_pred_list[current_node] = (s_pred)
            context_posterior_list.append(context)
            # if self.eq_model.family_dict is not None and current_node >= self.n_concepts:
            #     concept_group.append(current_node)
            #     s_preds_tmp.append(c_to_use)
            #     for key, value in self.eq_model.family_dict.items():
            #         if concept_group == value:
            #             concept_group = []
            #             s_preds_tmp = torch.cat(s_preds_tmp, axis=1)
            #             s_preds_tmp = torch.softmax(s_preds_tmp, dim=1)
            #             for i, el in enumerate(value):
            #                 s_pred_list[el] = s_preds_tmp[:, i].unsqueeze(1)
            #                 c_emb_true = self._build_concept_embedding(context, c_to_use)
            #                 if c_emb_dict == []:
            #                     c_emb_dict = torch.zeros(c_emb_true.shape[0], self.n_symbols, c_emb_true.shape[1])
            #                 c_emb_dict[:, el] = c_emb_true
            #             s_preds_tmp = []
            # else:
            # build the concept embedding of the current node using its context and the concept value predicted by its context if a root node or by the concept embeddings of its parents if not a root node
            c_emb_true = self._build_concept_embedding(context, c_to_use)
            if c_emb_dict == []:
                c_emb_dict = torch.zeros(c_emb_true.shape[0], self.n_symbols, c_emb_true.shape[1])
            c_emb_dict[:, current_node] = c_emb_true
        s_prior = torch.cat(s_pred_list, axis=1)
        s_preds = torch.cat(s_pred_list, axis=1)
        s_emb_prior = c_emb_dict
        s_emb_post = c_emb_dict
        
        return s_prior, s_preds, s_emb_prior, s_emb_post, context_posterior_list
    

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
    ):
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return c_true[:, concept_idx:concept_idx + 1] 


class CausalCGM(LightningModule):
    def __init__(self, input_dim, embedding_size, n_concepts, n_classes, ce_size, gamma, lambda_orth=0, lambda_cace=0, probabilistic=False, root_loss_l = 0.1,
                #  weight=None, family_of_concepts=None
                ):
        super().__init__()
        # encoder to embed the input data into a latent space used by all the concepts encoders
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )
        # Causal Concept Graph Model
        self.concept_embedder = CausalConceptGraphLayer(embedding_size, n_concepts, n_classes, embedding_size, gamma=gamma)

        # loss function
        self.loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()

        # parameters
        self.n_symbols = n_concepts + n_classes 
        self.lambda_orth = lambda_orth
        self.lambda_cace = lambda_cace
        self.root_loss_l = root_loss_l

    def forward(self, x, c=None, intervention_idxs=[], train=False):
        """
        Forward pass of the CausalCGM.
        """
        # embed the input data into a latent space used by all the concepts encoders
        emb = self.encoder(x)
        # forward pass of the CausalConceptGraphLayer
        output = self.concept_embedder.forward(emb, c=c, intervention_idxs=intervention_idxs, train=train)
        if train:
            return output
        else:
            c_pred, y_pred = output
        # concatenate the concept predictions and the class predictions
            s_pred = torch.cat([c_pred, y_pred], dim=-1)
            return s_pred

    def training_step(self, batch, batch_idx):
        x, c, y = batch
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        c_tmp = torch.cat([c, y], dim=-1)
        (c_preds_prior, y_preds_prior,
         c_preds_posterior, y_preds_posterior,
         s_preds_posterior0, s_preds_posterior1, s_emb_prior, s_emb_posterior, context_post) = self.forward(x, c=c_tmp, intervention_idxs=torch.arange(c_tmp.shape[1]), train=True)

        # compute the CACE loss - it is computing on average how many concepts and labels are you changing when you intervene on a random concept
        cace_loss = cace_score(s_preds_posterior1, s_preds_posterior0).norm()

        # compute the DAG
        self.concept_embedder.compute_parent_indices()

        # compute which are the root nodes
        is_root = self.concept_embedder.scm.sum(dim=0) == 0

        # compute the DAG loss
        dag_loss = self.concept_embedder.eq_model.h_func() 

        # compute the prior loss - it is the loss on the first level of concepts (c), as they would all be root nodes
        prior_loss = self.loss(c_preds_prior, c) + self.loss(y_preds_prior.squeeze(), y.squeeze())

        # compute the posterior loss - it is the loss on the second level of concepts - copies - (c'), as they would all be leaf nodes and they are predicted using their parents
        posterior_loss = self.loss(c_preds_posterior, c) + self.loss(y_preds_posterior.squeeze(), y.squeeze())

        loss = prior_loss + posterior_loss + 3*dag_loss + self.lambda_cace / (cace_loss + 1e-6)

        # compute the accuracy of the prior and posterior predictions
        prior_task_accuracy = accuracy_score(y.cpu().squeeze(), y_preds_prior.detach().cpu().squeeze() > 0.5)
        prior_concept_accuracy = accuracy_score(c.cpu(), c_preds_prior.detach().cpu() > 0.5)
        
        # compute the accuracy of the posterior predictions
        posterior_task_accuracy = accuracy_score(y.cpu().squeeze(), y_preds_posterior.detach().cpu().squeeze() > 0.5)
        posterior_concept_accuracy = accuracy_score(c.cpu(), c_preds_posterior.detach().cpu() > 0.5)

        # log the loss and the accuracy
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('dag', dag_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('cace', cace_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('c acc pri', prior_concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('c acc pos', posterior_concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('y acc pri', prior_task_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('y acc pos', posterior_task_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, c, y = batch
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        s = torch.cat([c, y], dim=-1)
        # forward pass of the CausalCGM
        s_pred = self.forward(x)
        # compute the prediction loss
        loss = self.loss(s_pred, s) 
        # compute the general accuracy
        concept_accuracy = accuracy_score(s.cpu().ravel(), s_pred.detach().cpu().ravel() > 0.5)
        # log the loss and the accuracy
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_concept_accuracy', concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, c, y = batch
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        s = torch.cat([c, y], dim=-1)
        # forward pass of the CausalCGM
        s_pred = self.forward(x)
        # compute the prediction loss
        loss = self.loss(s_pred, s) 
        # compute the general accuracy
        concept_accuracy = accuracy_score(s.cpu().ravel(), s_pred.detach().cpu().ravel() > 0.5)
        # log the loss and the accuracy
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_concept_accuracy', concept_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)