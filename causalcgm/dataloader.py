import os.path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
from causalcgm.utils import colorize
from causalcgm.utils import checkmark_dataset, preprocess_concept_celeba

def load_dag(dataset):
    """
    Load a ground truth or causal discovered DAG. 
    You can discover new causal graph using discover_grasp_dag() in utils.py
    Args:
        dataset: str
                 Name of the dataset
    Returns:
        dag_init: torch.Tensor, shape (number of concepts, number of concepts)
                  Adjacency matrix of the DAG
        to_check: list of tuples
                  List of edges to check
        concept_names: list of str
                       Names of the concepts
    """
    if dataset == "dsprites":
        dag_init = torch.FloatTensor([[0, 1, 0, 0, 1, 0], 
                            [0, 0, 0, 0, 0, 1], 
                            [0, 0, 0, 0, 1, 0], 
                            [0, 1, 0, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0]
                            ])
        to_check = []
        concept_names = ['Shape', 'Size', 'PosY', 'PosX', 'Color', 'Label']
        return dag_init, to_check, concept_names
    elif dataset == 'checkmark':
        dag_init = torch.FloatTensor([[0, 0, 0, 1],  # A influences D
                            [0, 0, 0, 0],  # B influences C
                            [0, 0, 0, 1],  # C influences D
                            [0, 0, 0, 0],  # D doesn't influence others
                            ])
        to_check = [(1,2)]
        concept_names = ['A', 'B', 'C', 'D']
        return dag_init, to_check, concept_names
    elif dataset == 'celeba':
        dag_init = torch.tensor([[0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.],
                                [0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1.],
                                [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1.],
                                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                [0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.],
                                [0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0.]])
        to_check = [(2, 0), (4, 0)]
        concept_names = ['Smiling', 'Attractive', 'Mouth_Slig', 'High_Cheek', 'Wearing_Li', 'Heavy_Make', 'Male', 'Wavy_Hair', 'Big_Lips', 'Oval_Face', 'Makeup', 'Fem_Model']
        return dag_init, to_check, concept_names

# Directly load embeddings computed with a backbone (e.g. ResNet18)
def load_preprocessed_data(base_dir='./dataset/dsprites'):
    """
    Load preprocessed data of a dataset.
    Args:
        base_dir: str
                  Path to the directory containing the preprocessed data
    Returns:
        train_features: torch.Tensor, shape (number of samples, feature dimension)
                        Features of the training set
        train_concepts: torch.Tensor, shape (number of samples, number of concepts)
                        Concept values of the training set
        train_tasks: torch.Tensor, shape (number of samples, 1)
                     Task labels of the training set
        test_features: torch.Tensor, shape (number of samples, feature dimension)
                       Features of the test set
        test_concepts: torch.Tensor, shape (number of samples, number of concepts)
                       Concept values of the test set
        test_tasks: torch.Tensor, shape (number of samples, 1)
                    Task labels of the test set
    """
    dag, to_check, concept_names = load_dag(base_dir.split('/')[-1])
    if 'checkmark' in base_dir:
        train_features, train_concepts, train_tasks = checkmark_dataset(800, 42, 0.2, return_y=True)
        test_features, test_concepts, test_tasks = checkmark_dataset(200, 24, 0.2, return_y=True)
        
    # Load training
    else:
        train_features = torch.from_numpy(np.load(os.path.join(base_dir, 'train_features.npy')))
        train_concepts = torch.from_numpy(np.load(os.path.join(base_dir, 'train_concepts.npy')))
        train_tasks = torch.from_numpy(np.load(os.path.join(base_dir, 'train_tasks.npy'))).unsqueeze(1).float()
        # Load test
        test_features = torch.from_numpy(np.load(os.path.join(base_dir, 'test_features.npy')))
        test_concepts = torch.from_numpy(np.load(os.path.join(base_dir, 'test_concepts.npy')))
        test_tasks = torch.from_numpy(np.load(os.path.join(base_dir, 'test_tasks.npy'))).unsqueeze(1).float()
        if 'celeba' in base_dir:
            order = [29, -3, 19, 18, 34, 17, -2, 31, 5, 23]
            train_features, train_concepts, concept_names, _ = preprocess_concept_celeba(train_features, train_concepts, train_tasks, order=order)
            test_features, test_concepts, tmp2, _ = preprocess_concept_celeba(test_features, test_concepts, test_tasks, order=order)
            train_tasks = train_concepts[:, -1].unsqueeze(-1)
            test_tasks = test_concepts[:, -1].unsqueeze(-1)
            train_concepts = train_concepts[:, :-1]
            test_concepts = test_concepts[:, :-1]
    
    return (train_features, train_concepts, train_tasks, 
            test_features, test_concepts, test_tasks,
            dag, to_check, concept_names)



