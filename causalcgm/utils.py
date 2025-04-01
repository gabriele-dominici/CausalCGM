import os.path
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import accuracy_score
from torchvision.datasets import CelebA
from pathlib import Path
import torch.nn.functional as F
from causallearn.search.PermutationBased.GRaSP import grasp


# Function Utils

# CaCe Score (https://arxiv.org/pdf/1907.07165)
def cace_score(c_pred_c0, c_pred_c1):
    """
    Computes the CACE score.
    Args:
        c_pred_c0: torch.Tensor, shape (batch_size, number of concepts)
                   concept values where we did do intervention making a concept equal to 0
        c_pred_c1: torch.Tensor, shape (batch_size, number of concepts)
                   concept values where we did do intervention making a concept equal to 1
    Returns:
        cace: torch.Tensor, shape (number of concepts)
    """
    cace = torch.abs(c_pred_c1.mean(dim=0) - c_pred_c0.mean(dim=0))
    return cace

# Entropy
def entropy(Y):
    """
    Also known as Shanon Entropy
    Reference: https://en.wikipedia.org/wiki/Entropy_(information_theory)
    Args:
        Y: numpy array, shape (number of samples, number of random variables)
           Random variables
    Returns:
        en: float
            Entropy of the random variables
    """
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count/len(Y)
    en = np.sum((-1)*prob*np.log2(prob))
    return en


#Joint Entropy
def jEntropy(Y,X):
    """
    H(Y;X)
    Reference: https://en.wikipedia.org/wiki/Joint_entropy
    Args:
        Y: numpy array, shape (number of samples, number of random variables)
           Random variables
        X: numpy array, shape (number of samples, number of random variables)
           Random variables
    Returns:
        en: float
            Joint Entropy of the random variables
    """
    YX = np.c_[Y,X]
    return entropy(YX)

#Conditional Entropy
def cEntropy(Y, X):
    """
    conditional entropy = Joint Entropy - Entropy of X
    H(Y|X) = H(Y;X) - H(X)
    Reference: https://en.wikipedia.org/wiki/Conditional_entropy
    Args:
        Y: numpy array, shape (number of samples, number of random variables)
           Random variables
        X: numpy array, shape (number of samples, number of random variables)
           Random variables
    Returns:
        en: float
            Conditional Entropy of the random variables
    """
    return jEntropy(Y, X) - entropy(X)

# Conditional entropy values among all concepts
def conditional_entropy_dag(s):
    """
    Compute the conditional entropy values among all concepts and tasks.
    Args:
        s: numpy array, shape (number of samples, number of concepts + number of tasks)
           Concept and task values
    Returns:
        dag: numpy array, shape (number of concepts + number of tasks, number of concepts + number of tasks)
             Conditional entropy values among all concepts and tasks
    """
    dag = np.zeros((s.shape[1], s.shape[1]))
    for i in range(s.shape[1]):
        for j in range(s.shape[1]):
            if i != j:
                dag[i, j] = 1 - cEntropy(s[:, i], s[:, j])
    return dag

# Compute the Probability of Necessity and Sufficiency Matrix using CausalCGM
def compute_pns_matrix(x, model, dag):
    """
    Compute the Probability of Necessity and Sufficiency Matrix using CausalCGM.
    Args:
        x: torch.Tensor, shape (number of samples, input dimension)
           Input features
        model: torch.nn.Module
               CausalCGM model
        dag: numpy array, shape (number of concepts + number of tasks, number of concepts + number of tasks)
             Directed acyclic graph (DAG) representing the causal relationships among concepts and tasks learnt by CausalCGM
    Returns:
        matrix_pns: numpy array, shape (number of concepts + number of tasks, number of concepts + number of tasks, 2) 2 represents the min and max values of PNS
                    Probability of Necessity and Sufficiency Matrix
    """ 
    #create a graph from the adjacency matrix
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    # fill dag with ones where there is an indirected connection
    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if i != j:
                if nx.has_path(G, i, j):
                        dag[i, j] = 1
    # matrix nxn zeros
    matrix_pns = np.zeros((model.n_symbols, model.n_symbols))
    matrix_pns[:] = np.nan
    matrix_pns = matrix_pns.tolist()
    zero = torch.zeros((x.shape[0], model.n_symbols))
    one = torch.ones((x.shape[0], model.n_symbols))
    for i in range(model.n_symbols):
        # get the prediction of the model with the intervention both to 0 and 1
        s_pred_c1 = (model.forward(x, c=one, intervention_idxs=[i]) > 0.5).float()
        s_pred_c0 = (model.forward(x, c=zero, intervention_idxs=[i]) > 0.5).float()

        # get the probability of all the other values when the intervention is 0 and 1
        # P(y|do(c_i=0))
        py_notx = s_pred_c0.mean(dim=0)
        # P(y|do(c_i=1))
        py_x = s_pred_c1.mean(dim=0)
        # P(~y|do(c_i=0)) = 1 - P(y|do(c_i=0))
        pnoty_notx = 1 - py_notx
        
        row = dag[i]

        for j, el in enumerate(row):
            if el != 0:
                # Lower bound of PNS PNS_min = max(0, P(y|do(c_i=1)) - P(y|do(c_i=0)))
                pns_min = max((py_x[j] - py_notx[j]).item(), 0)
                # Upper bound of PNS PNS_max = min(P(y|do(c_i=1)), P(~y|do(c_i=0)))
                pns_max = min(py_x[j].item(), pnoty_notx[j].item())
                matrix_pns[i][j] = (np.around(pns_min, 2), np.around(pns_max, 2))
    return matrix_pns

# Interventions (do-interventions) starting from root nodes
def interventions_from_root(dag, model, x, s, order=None, exclude=None, exclude_labels=[], compute_label=False, absolute=False):
    """
    Perform interventions starting from root nodes.
    Args:
        dag: numpy array, shape (number of concepts + number of tasks, number of concepts + number of tasks)
             Directed acyclic graph (DAG) representing the causal relationships among concepts and tasks learnt by CausalCGM
        model: torch.nn.Module
               CausalCGM model
        x: torch.Tensor, shape (number of samples, input dimension)
           Input features
        s: torch.Tensor, shape (number of samples, number of concepts + number of tasks)
           Concept and task values
        order: list
               Order of the interventions if we want to force an order
        exclude: list
                 List of concepts to exclude from interventions 
    Returns:
        acc: list
             List of changes in accuracy of the concepts and tasks after the interventions
        abs_acc: list
                 List of absolute changes in accuracy of the concepts and tasks after the interventions 
    """
    if exclude is None:
        # not intervene on the task (in our experiments always the last one)
        exclude = [s.shape[1]-1]
    # binarize the graph if needed
    dag = (dag > 0.1).float().numpy()
    G = nx.from_numpy_array(dag, create_using=nx.DiGraph)
    # # fill dag with ones where there is an indirected connection
    for i in range(dag.shape[0]):
        for j in range(dag.shape[1]):
            if i != j:
                if nx.has_path(G, i, j):
                        dag[i, j] = 1
    # get the number of indirected connections for each node
    connections = dag.sum(axis=1)
    # order the nodes by the number of connections (highest first)
    order_connections = np.flip(np.argsort(connections))
    # get the parent indices of each node through a topological sort (alternative order)
    parent_indices = {node: list(G.predecessors(node)) for node in list(nx.topological_sort(G))}
    
    # add random noise to x to decrease the performance of the model
    x_perturbed = x.clone()
    x_perturbed = x_perturbed + torch.randn_like(x_perturbed) * 15
    # for i in range(s.shape[1]):
    #     # get the accuracy of the model on the original perturbed data
    #     s_pred = model(x_perturbed)
    #     concept_accuracy = accuracy_score(s[:, i].ravel(), s_pred[:, i].ravel() > 0.5)
    
    # get the accuracy of the model on the original perturbed data
    int_idexes = []
    s_pred = model(x_perturbed)
    if compute_label:
        to_include = [s.shape[1]-1]
    else:
        to_include = [i for i in range(s.shape[1]) if i not in int_idexes and i not in exclude]
    if absolute:
        to_include = [i for i in range(s.shape[1])]
    concept_accuracy = accuracy_score(s[:, to_include].ravel(), s_pred[:, to_include].ravel() > 0.5)
    acc = []
    abs_acc = [concept_accuracy]
    if order is None:
        for current_node in order_connections:
            if current_node in exclude:
                continue
            elif current_node in exclude_labels:
                continue
            else:
                # update the intervention indexes
                int_idexes += [current_node]
                # get the predictions of the model with the intervention
                s_pred_int = model(x_perturbed, c=s, intervention_idxs=int_idexes)
                # get the predictions of the model without the intervention
                s_pred = model(x_perturbed)
                # get the concepts to include in the accuracy computation (no nodes intervened and no nodes excluded)
                if compute_label:
                    to_include = [s.shape[1]-1]
                else:
                    to_include = [i for i in range(s.shape[1]) if i not in int_idexes and i not in exclude]
                if absolute:
                    to_include = [i for i in range(s.shape[1])]
                if to_include != []:
                    # get the accuracy of the model without the intervention on the concepts to include
                    concept_accuracy = accuracy_score(s[:, to_include].ravel(), s_pred[:, to_include].ravel() > 0.5)
                    print(concept_accuracy)
                    # get the accuracy of the model with the intervention on the concepts to include
                    concept_accuracy_int = accuracy_score(s[:, to_include].ravel(), s_pred_int[:, to_include].ravel() > 0.5)
                    print(concept_accuracy_int)
                    # get the change in accuracy
                    acc += [concept_accuracy_int-concept_accuracy]
                    # get the absolute accuracy
                    # concept_accuracy_abs = accuracy_score(s[:, to_include].ravel(), s_pred_int[:, to_include].ravel() > 0.5)
                    abs_acc += [concept_accuracy_int]
    else:
        # Used for baseline where the order is random
        for current_node in order:
            if current_node in exclude:
                continue
            if current_node in exclude_labels:
                continue
            else:
                # update the intervention indexes
                int_idexes += [current_node]
                # get the predictions of the model with the intervention
                s_pred_int = model(x_perturbed, c=s, intervention_idxs=int_idexes)
                # get the predictions of the model without the intervention
                s_pred = model(x_perturbed)
                # get the concepts to include in the accuracy computation (no nodes intervened and no nodes excluded)
                if compute_label:
                    to_include = [s.shape[1]-1]
                else:
                    to_include = [i for i in range(s.shape[1]) if i not in int_idexes and i not in exclude]
                if absolute:
                    to_include = [i for i in range(s.shape[1])]
                if to_include != []:
                    # get the accuracy of the model without the intervention on the concepts to include
                    concept_accuracy = accuracy_score(s[:, to_include].ravel(), s_pred[:, to_include].ravel() > 0.5)
                    # get the accuracy of the model with the intervention on the concepts to include
                    concept_accuracy_int = accuracy_score(s[:, to_include].ravel(), s_pred_int[:, to_include].ravel() > 0.5)
                    # get the change in accuracy
                    print(concept_accuracy, concept_accuracy_int)
                    acc += [concept_accuracy_int-concept_accuracy]
                    # get the absolute accuracy
                    # concept_accuracy_abs = accuracy_score(s[:, to_include].ravel(), s_pred_int[:, to_include].ravel() > 0.5)
                    abs_acc += [concept_accuracy_int]
    return acc, abs_acc

def discover_grasp_dag(s):
    """
    Discover the causal graph using GRaSP.
    Args:
        s: torch.Tensor, shape (number of samples, number of concepts)
           Label values
    Returns:
        adj: numpy array, shape (number of concepts, number of concepts)
             Adjacency matrix of the causal graph
        to_check: list
             List of edges to check
    """
    # extract the causal graph using GRaSP
    G = grasp(s.numpy(), score_func='local_score_BDeu')
    # parse the graph selecting the edges with a direction and the edges to check (no direction fixed)
    adj = np.zeros(G.graph.shape)
    to_check = []
    for j in range(G.graph.shape[0]):
        for i in range(G.graph.shape[1]):
            if G.graph[j, i] == 1 and G.graph[i, j] == -1:
                adj[i, j] = 1
            if G.graph[j, i] == -1 and G.graph[i, j] == -1:
                # adj[i, j] = 1
                # adj[j, i] = 1
                if (i, j) not in to_check and (j, i) not in to_check:
                    to_check.append((i, j))
    return adj, to_check


# Dataset Utils

# Helper function to convert grayscale image to red or green
def colorize(image, color):
    """
    Colorize a grayscale image to red, green or blue.
    Args:
        image: torch.Tensor, shape (1, 64, 64)
               Grayscale image
        color: str, 'red', 'green' or 'blue'
               Color to use for the image
    Returns:
        colored_image: torch.Tensor, shape (3, 64, 64)
    """
    colored_image = torch.zeros(3, 64, 64)  # Create an image with 3 channels (RGB)
    if color == 'red':
        colored_image[0] = image  # Red channel
    elif color == 'green':
        colored_image[1] = image  # Green channel
    elif color == 'blue':
        colored_image[2] = image  # Green channel
    return colored_image

# Custom CelebA dataset
class CelebADataset(CelebA):
    def __init__(self, root, split='train', transform=None, download=False, class_attributes=None):
        super(CelebADataset, self).__init__(root, split=split, target_type="attr", transform=transform, download=download)

        # Set the class attributes
        if class_attributes is None:
            # Default to 'Attractive' if no class_attributes provided
            self.class_idx = [self.attr_names.index('Attractive')]
        else:
            # Use the provided class attributes
            self.class_idx = [self.attr_names.index(attr) for attr in class_attributes]

        self.attr_names = [string for string in self.attr_names if string]

        # Determine concept and task attribute names based on class attributes
        self.concept_attr_names = [attr for i, attr in enumerate(self.attr_names) if i not in self.class_idx]
        self.task_attr_names = [self.attr_names[i] for i in self.class_idx]

    def __getitem__(self, index):
        image, attributes = super(CelebADataset, self).__getitem__(index)

        # Extract the target (y) based on the class index
        y = torch.stack([attributes[i] for i in self.class_idx])

        # Extract concept attributes, excluding the class attributes
        concept_attributes = torch.stack([attributes[i] for i in range(len(attributes)) if i not in self.class_idx])

        return image, concept_attributes, y

# Custom Resnet model for feature extraction
class ResNetEmbedding(torch.nn.Module):
    def __init__(self, original_model):
        super(ResNetEmbedding, self).__init__()
        self.features = torch.nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


# Creating the custom dataset
class CustomDSpritesDataset(torch.utils.data.Dataset):
    """
    Custom dataset for dSprites dataset.
    Args:
        dsprites_dataset: torch.Tensor, shape (number of samples, 1, 64, 64)
                          Images of the dSprites dataset
        concept_label: torch.Tensor, shape (number of samples, number of concepts)
                      Concept values of the dSprites dataset
        target_label: torch.Tensor, shape (number of samples, 1)
                      Target labels of the dSprites dataset
    """
    def __init__(self, dsprites_dataset, concept_label, target_label):
        self.dsprites_dataset = dsprites_dataset
        self.concept_label = concept_label
        self.target_label = target_label

    def __len__(self):
        return len(self.dsprites_dataset)

    def __getitem__(self, idx):
        image = self.dsprites_dataset[idx]
        concept_label = self.concept_label[idx]
        target_label = self.target_label[idx]

        # Colorize the image according to the concept value
        if concept_label[4] == 1:
            color = 'green'
        else:
            color = 'red'
        colored_image = colorize(image.squeeze(), color)  # Remove channel dimension of the grayscale image

        return colored_image, torch.tensor(concept_label, dtype=torch.float32), torch.tensor(target_label, dtype=torch.float32)
    
# Preprocess dataset
def preprocess_dataset(dataset_name="dsprites"):
    """
    Preprocess the dataset to extract features using a pretrained ResNet18 model.
    Args:
        dataset_name: str, ['dsprites', celeba']
                      Name of the dataset
    """
    # Step 1: Prepare the dataset
    # Create custom datasets
    if dataset_name == 'dsprites':
        # Load  dataset
        dsprites_train_img = torch.from_numpy(np.load(f'./datasets/{dataset_name}/train_images.npy'))
        dsprites_train_concepts = torch.from_numpy(np.load(f'./datasets/{dataset_name}/train_concepts.npy'))
        dsprites_train_labels = torch.from_numpy(np.load(f'./datasets/{dataset_name}/train_labels.npy'))
        dsprites_test_img = torch.from_numpy(np.load(f'./datasets/{dataset_name}/test_images.npy'))
        dsprites_test_concepts = torch.from_numpy(np.load(f'./datasets/{dataset_name}/test_concepts.npy'))
        dsprites_test_labels = torch.from_numpy(np.load(f'./datasets/{dataset_name}/test_labels.npy'))

        custom_train_dataset = CustomDSpritesDataset(dsprites_train_img, dsprites_train_concepts, dsprites_train_labels)
        custom_test_dataset = CustomDSpritesDataset(dsprites_test_img, dsprites_test_concepts, dsprites_test_labels)
    elif dataset_name == 'celeba':
        # Define image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        class_attributes = ['Attractive', 'Male', 'Young']

        # Load CelebA dataset
        custom_train_dataset = CelebADataset(root=f'./datasets/{dataset_name}/', split="train", transform=transform, download=False,
                                    class_attributes=class_attributes)
        custom_test_dataset = CelebADataset(root=f'./datasets/{dataset_name}/', split="test", transform=transform, download=False,
                                    class_attributes=class_attributes)
        torch.save(custom_train_dataset.concept_attr_names, f'./datasets/{dataset_name}/concept_names.pt')
        torch.save(custom_train_dataset.task_attr_names, f'./datasets/{dataset_name}/task_names.pt')

    # DataLoaders
    train_loader = DataLoader(custom_train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(custom_test_dataset, batch_size=512, shuffle=False)

    # Step 2: Prepare ResNet18 model for feature extraction
    model_resnet = models.resnet18(pretrained=True)
    model = ResNetEmbedding(model_resnet)
    model.eval()  # Set the model to evaluation mode

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Step 3: Extract features
    def extract_features(data_loader):
        features = []
        concept_labels = []
        task_labels = []

        with torch.no_grad():
            for imgs, concepts, tasks in tqdm(data_loader):
                imgs = imgs.to(device)
                out = model(imgs)
                features.append(out.cpu().numpy())
                concept_labels.append(concepts.numpy())
                task_labels.append(tasks.numpy())

        return np.concatenate(features), np.concatenate(concept_labels), np.concatenate(task_labels)

    train_features, train_concepts, train_tasks = extract_features(train_loader)
    test_features, test_concepts, test_tasks = extract_features(test_loader)

    # Step 4: Save the embeddings and labels (concept and task)
    np.save(f'./datasets/{dataset_name}/train_features.npy', train_features)
    np.save(f'./datasets/{dataset_name}/train_concepts.npy', train_concepts)
    np.save(f'./datasets/{dataset_name}/train_tasks.npy', train_tasks)

    np.save(f'./datasets/{dataset_name}/test_features.npy', test_features)
    np.save(f'./datasets/{dataset_name}/test_concepts.npy', test_concepts)
    np.save(f'./datasets/{dataset_name}/test_tasks.npy', test_tasks)

def toy_problem(n_samples=10, seed=42):
    """
    Create a toy problem with 4 columns (A, B, C, D) and n_samples samples.
    Args:
        n_samples: int
                   Number of samples
        seed: int
              Random seed
    Returns:
        x: torch.Tensor, shape (n_samples, 4)
           Toy problem dataset
    """
    torch.manual_seed(seed)
    A = torch.randint(0, 2, (n_samples,), dtype=torch.bool)
    torch.manual_seed(seed + 1)
    B = torch.randint(0, 2, (n_samples,), dtype=torch.bool)

    # Column C is true if B is true, randomly true/false if B is false
    C = ~B

    # Column D is true if A or C is true, randomly true/false if both are false
    D = A & C

    # Combine all columns into a matrix
    return torch.stack((A, B, C, D), dim=1).float()


def checkmark_dataset(n_samples=800, seed=42, perturb=0.1, return_y=False):
    """
    Create a checkmark dataset, starting from the toy problem.
    Args:
        n_samples: int
                   Number of samples
        seed: int
              Random seed
        perturb: float
                 Standard deviation of the noise
        return_y: bool
                  Return the target labels or not
    Returns:
        x: torch.Tensor, shape (n_samples, 4)
           Numbers connected to the concept values
        c: torch.Tensor, shape (n_samples, 3)
           Concept values
        y: torch.Tensor, shape (n_samples, 1)
           Target labels
    """
    x = toy_problem(n_samples, seed)
    c = x.clone()
    torch.manual_seed(seed)
    x = x * 2 - 1 + torch.randn_like(x) * perturb

    if return_y:
        return x, c[:, [0, 1, 2]], c[:, 3].unsqueeze(1)
    else:
        return x, c


def preprocess_concept_celeba(embeddings, concepts, tasks, order=None):
    """
    Filter the concepts and tasks of the CelebA dataset, while also create new intermediate concepts.
    Args:
        embeddings: torch.Tensor, shape (number of samples, feature dimension)
                Image embeddings of the dataset
        concepts: torch.Tensor, shape (number of samples, number of concepts)
                Concept values of the dataset
        tasks: torch.Tensor, shape (number of samples, 1)
               Task labels of the dataset
        order: list
               Order of the concepts to filter
    """
    # Load the concept names
    concept_names = torch.load('./datasets/celeba/concept_names.pt')
    # Load the task names
    task_names = torch.load('./datasets/celeba/task_names.pt')

    # Combine concepts and tasks as they are the same for us
    labels = torch.cat((concepts, tasks.squeeze()), dim=1).float()
    label_names = concept_names + task_names

    # Count the number of 1s in each column
    sums = labels.sum(dim=0)
    # Calculate the balance score for each column
    balance_scores = torch.abs(sums - labels.size(0) / 2)
    # Rank columns by their balance score (lower is more equilibrated)
    ranked_columns = torch.argsort(balance_scores)[:10]
    # If an order is provided, use it
    if order is not None:
        ranked_columns = order
    ranked_label_names = [label_names[i] for i in ranked_columns]
    ranked_labels = labels[:, ranked_columns]

    # intermediate concepts
    ci1_id1 = ranked_label_names.index('Wearing_Lipstick')
    ci1_id2 = ranked_label_names.index('Heavy_Makeup')
    # ci1_id3 = ranked_label_names.index('Big_Lips')
    ci1 = concepts[:, ci1_id1] & concepts[:, ci1_id2] #| concepts[:, ci1_id3]
    labeli1 = 'Makeup'
    ci2_id1 = ranked_label_names.index('Attractive')
    ci2_id2 = ranked_label_names.index('Male')
    ci2 = (concepts[:, ci2_id1] | ci1) & ~concepts[:, ci2_id2]
    labeli2 = 'Fem_Model'

    ranked_labels = torch.cat((ranked_labels, ci1.unsqueeze(1), ci2.unsqueeze(1)), dim=1)
    ranked_label_names = ranked_label_names + [labeli1, labeli2]

    print(ranked_labels.mean(dim=0))

    # ranked_labels = torch.cat([ranked_labels[:, ranked_label_names.index(el)].unsqueeze(1) for el in ranked_label_names if el != 'Attractive'] + 
    #                           [ranked_labels[:, ranked_label_names.index('Attractive')].unsqueeze(1)], dim=1)
    # # ranked_labels = torch.cat([ranked_labels, ], dim=1)
    # print(ranked_labels.mean(dim=0))

    # ranked_label_names = [i for i in ranked_label_names if i != 'Attractive']
    # ranked_label_names = ranked_label_names + ['Attractive']

    ranked_label_names = [ln[:10] for ln in ranked_label_names]
    print(ranked_label_names)
    return embeddings, ranked_labels, ranked_label_names, ranked_columns