import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer, seed_everything
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from causalcgm.utils import cace_score, interventions_from_root, compute_pns_matrix, conditional_entropy_dag
import numpy as np
import os
from causalcgm.causalcgm import CausalCGM
from causalcgm.baselines import CBM, CEM, StandardE2E
import pytorch_lightning as pl
from causalcgm.dataloader import load_preprocessed_data
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = ''

seed_everything(0)

n_seeds = 5
embedding_size = 8
ce_size = embedding_size 
gamma = 1

# ATTENTION WITH THIS: They are correct for the given graphs
# they are not general (if you learn a graph you should change this according to the learned graph)
index_perturb = {'dsprites': [3], 'checkmark': [1], 'celeba': [2]} # should be all parents of the index block
index_block = {'dsprites': [1], 'checkmark': [2], 'celeba': [0, 1]} # should be the index of the children of the index perturb and should be an ancestor of the label (last concept in the graph)

datasets = ['checkmark', 'dsprites', 'celeba']
epochs = {'dsprites': 200, 'checkmark': 350, 'celeba': 40}
no_out_task = {'dsprites': True, 'checkmark': True, 'celeba': False}

random_start = False # if True, the model starts with a random DAG, otherwise it starts with a DAG initialized with the conditional entropy between concepts

for dataset in datasets:
    results_dir = f'./results/{dataset}'
    os.makedirs(results_dir, exist_ok=True)
    results = []
    interventions = []
    interventions_c = []
    explanations = []
    interventions_y = []
    interventions_abs = []
    pns_list = []
    for i in range(n_seeds):
        seed = i
        seed_everything(seed)
        lambda_cace_2 = 0.0

        (x_train, c_train, y_train, 
        x_test, c_test, y_test,
        dag_init, to_check, label_names) = load_preprocessed_data(base_dir=f'./datasets/{dataset}')
        n = x_train.shape[0]
        # random shuffle train
        idx = torch.randperm(n)
        x_train = x_train[idx]
        c_train = c_train[idx]
        y_train = y_train[idx]
        x_val = x_train[-int(n*0.2):]
        c_val = c_train[-int(n*0.2):]
        y_val = y_train[-int(n*0.2):]
        x_train = x_train[:int(n*0.8)]
        c_train = c_train[:int(n*0.8)]
        y_train = y_train[:int(n*0.8)]
        n_concepts = c_train.shape[1]
        n_classes = y_train.shape[1]
        s_train = torch.cat((c_train, y_train), dim=1)
        s_val = torch.cat((c_val, y_val), dim=1)
        s_test = torch.cat((c_test, y_test), dim=1)
        
        train_loader = DataLoader(TensorDataset(x_train, s_train), batch_size=128, shuffle=True)
        train_loader_cbm = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=128, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, s_val), batch_size=128, shuffle=False)
        val_loader_cbm = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=128, shuffle=False)
        test_loader = DataLoader(TensorDataset(x_test, s_test), batch_size=128, shuffle=False)
        test_loader_cbm = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=128, shuffle=False)
        n_symbols = s_train.shape[1]
        n_concepts = c_train.shape[1]
        n_classes = y_train.shape[1]
        tp_size = n_concepts*ce_size

        models = {

            'CausalCGM': CausalCGM(x_train.shape[1], embedding_size, n_concepts, n_classes, ce_size, gamma, 0, lambda_cace_2, probabilistic=False, no_out_task=no_out_task[dataset]),
            'CausalCGM_given': CausalCGM(x_train.shape[1], embedding_size, n_concepts, n_classes, ce_size, gamma, 0, lambda_cace_2, probabilistic=False, no_out_task=no_out_task[dataset]),
            'CEM': CEM(x_train.shape[1], embedding_size, n_concepts, n_classes, ce_size, tp_size),
            'CBM': CBM(x_train.shape[1], embedding_size, n_concepts, n_classes),
            # 'BB': StandardE2E(x_train.shape[1], n_classes+n_concepts, emb_size=embedding_size),
        }

        for model_name, model in models.items():
            # checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
            print(f'Training {model_name} with seed {seed+1}/{n_seeds}...')

            checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_concept_accuracy", mode="max", save_weights_only=True)
            if model_name.endswith('given') and dag_init is not None:
                model.concept_embedder.eq_model.fc1.weight = torch.nn.Parameter(dag_init, requires_grad=False)
                model.concept_embedder.eq_model.edges_to_check = to_check
            elif model_name not in ['CEM', 'CBM', 'BB']:
                if not random_start:
                # cov = torch.abs(torch.tensor(np.corrcoef(s_train.T))).float()/2
                    cov = conditional_entropy_dag(s_train)
                    cov = torch.tensor(cov).float()
                    cov[-1, :] = 0
                    cov = torch.clamp((cov / cov.mean()), 0, 0.99)
                    model.concept_embedder.eq_model.fc1.weight = torch.nn.Parameter(cov, requires_grad=True)
            # try:
            trainer = Trainer(max_epochs=epochs[dataset], accelerator='cpu', enable_checkpointing=True, callbacks=checkpoint_callback)
            trainer.fit(model, train_loader_cbm, val_loader_cbm)
            # except:
            #     continue
            model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
            model.eval()

            # data we need to test models
            c_test = s_test[:, :-1].clone()
            y_test = s_test[:, -1]
            s_fake_1 = s_test.clone()
            s_fake_0 = s_test.clone()
            for i_p in index_perturb[dataset]:
                s_fake_1[:, [i_p]] = 1
                s_fake_0[:, [i_p]] = 0
            c_fake_1 = s_fake_1[:, :-1]
            c_fake_0 = s_fake_0[:, :-1]
            s_perturb = s_test.clone()
            for i_p in index_perturb[dataset]:
                s_perturb[:, [i_p]] = 1 - s_test[:, [i_p]]
            c_perturb = s_perturb[:, :-1]

            if model.__class__.__name__ in ['CausalCGM']:
                # plot DAG
                # if model.__class__.__name__ == 'CausalCBM':
                model.concept_embedder.compute_parent_indices()
                dag = model.concept_embedder.dag
                print(model.concept_embedder.eq_model.fc1_to_adj())
                print(dag)
                # dag = model.concept_embedder.eq_model.fc1_to_adj().detach().cpu().numpy()
                plt.figure(figsize=(4, 4))
                plt.title(f'{dataset} DAG')
                dag_tmp = (dag > 0.01).astype(float)
                sns.heatmap(dag_tmp, xticklabels=label_names, yticklabels=label_names, cmap='coolwarm', cbar=True, vmin=0, vmax=1)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f'{dataset}_heatmap_{model_name}_{seed}.pdf'))
                # plt.show()

                # compute standard accuracy
                # do interventions without blocking
                s_pred_c1 = model.forward(x_test, c=s_fake_1, intervention_idxs=[*index_perturb[dataset]])
                s_pred_c0 = model.forward(x_test, c=s_fake_0, intervention_idxs=[*index_perturb[dataset]])

                # do interventions with blocking
                s_pred_c1_block = model.forward(x_test, c=s_fake_1, intervention_idxs=[*index_perturb[dataset], *index_block[dataset]])
                s_pred_c0_block = model.forward(x_test, c=s_fake_0, intervention_idxs=[*index_perturb[dataset], *index_block[dataset]])

                # do perturbation without blocking
                s_pred_perturb = model.forward(x_test, c=s_perturb, intervention_idxs=[*index_perturb[dataset]])

                # do perturbation with blocking
                s_pred_perturb_block = model.forward(x_test, c=s_perturb, intervention_idxs=[*index_perturb[dataset], *index_block[dataset]])


                s_pred = model.forward(x_test)

                # interventions
                acc_int = interventions_from_root(torch.tensor(dag), model, x_test, s_test, exclude=[], exclude_labels=[n_symbols-1])
                acc_int_only_concept = interventions_from_root(torch.tensor(dag), model, x_test, s_test, exclude=[n_symbols-1], exclude_labels=[n_symbols-1])
                acc_int_abs = interventions_from_root(torch.tensor(dag), model, x_test, s_test, exclude=[], exclude_labels=[n_symbols-1], absolute=True)
                
                # Create an order where parents of the label come first, then random order
                parent_indices = np.where(dag[:n_concepts, -1] > 0.05)[0].tolist()
                # order parent_indices according to their value in the dag
                parent_indices = sorted(parent_indices, key=lambda x: dag[x, -1], reverse=True)
                remaining_indices = [i for i in range(n_concepts) if i not in parent_indices]
                np.random.shuffle(remaining_indices)
                order = parent_indices + remaining_indices

                acc_int_only_label = interventions_from_root(torch.tensor(dag), model, x_test, s_test, exclude=[], exclude_labels=[n_symbols-1], order=order, compute_label=True)
                # PNS
                PNS = compute_pns_matrix(x_test, model, dag)

            elif model_name in ['CEM', 'CBM']:
                # compute standard accuracy
                s_pred = model.forward(x_test)

                # interventions
                order = torch.randperm(n_concepts).numpy().tolist() # random order of concepts
                acc_int = interventions_from_root(torch.tensor(dag_init), model, x_test, s_test, exclude=[], exclude_labels=[n_symbols-1])
                acc_int_abs = interventions_from_root(torch.tensor(dag_init), model, x_test, s_test, exclude=[], exclude_labels=[n_symbols-1], absolute=True)
                
                acc_int_only_concept = interventions_from_root(torch.tensor(dag_init), model, x_test, s_test, exclude=[n_symbols-1], exclude_labels=[n_symbols-1])
                acc_int_only_label = interventions_from_root(torch.tensor(dag_init), model, x_test, s_test, exclude=[], exclude_labels=[n_symbols-1], compute_label=True)
                # PNS
                dag = torch.zeros(n_symbols, n_symbols)
                dag[:, -1] = 1
                dag[-1, -1] = 0
                PNS = compute_pns_matrix(x_test, model, dag.numpy())

                # do interventions without blocking
                s_pred_c1 = model.forward(x_test, c=c_fake_1, intervention_idxs=[*index_perturb[dataset]], train=False)
                s_pred_c0 = model.forward(x_test, c=c_fake_0, intervention_idxs=[*index_perturb[dataset]], train=False)

                # do interventions with blocking
                s_pred_c1_block = model.forward(x_test, c=c_fake_1, intervention_idxs=[*index_perturb[dataset], *index_block[dataset]], train=False)
                s_pred_c0_block = model.forward(x_test, c=c_fake_0, intervention_idxs=[*index_perturb[dataset], *index_block[dataset]], train=False)

                # do perturbation without blocking
                s_pred_perturb = model.forward(x_test, c=c_perturb, intervention_idxs=[*index_perturb[dataset]], train=False)

                # do perturbation with blocking
                s_pred_perturb_block = model.forward(x_test, c=c_perturb, intervention_idxs=[*index_perturb[dataset], *index_block[dataset]], train=False)

            # compute metrics
            # s_pred = torch.cat([c_pred, y_pred], dim=1)
            # s_test = torch.cat([c_test, y_test], dim=1)
            if model_name == 'BB':
                s_pred = model.forward(x_test)
                s_accuracy = accuracy_score(s_test.ravel(), s_pred.ravel() > 0)
                print(f'{model_name} label accuracy: {s_accuracy}')
                # save results
                results.append([f'{dataset}_dataset', model_name, seed, s_accuracy])
                metric_cols = ['label_accuracy']
                df = pd.DataFrame(results, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'results_raw.csv'), index=False)
                df = df.drop(columns=['dataset'])
                mean_scaled = df.groupby('model').agg(lambda x: np.mean(x)) * 100
                variance_scaled = df.groupby('model').agg(lambda x: np.std(x)) * 100
                formatted_dfs = []
                for model in mean_scaled.index:
                    formatted_dfs.append(pd.DataFrame({col: f"${mean_scaled.loc[model, col]:.2f} \pm {variance_scaled.loc[model, col]:.2f}$" for col in metric_cols}, index=[model]))
                formatted_df = pd.concat(formatted_dfs)
                print(formatted_df.to_string(index=True, float_format='%.2f'))
                formatted_df.to_csv(os.path.join(results_dir, 'intv_results_table.csv'))
            else:
                s_accuracy = accuracy_score(s_test.ravel(), s_pred.ravel() > 0.5)
                print(f'{model_name} label accuracy: {s_accuracy}')

                cace = cace_score(s_pred_c1[:, -1], s_pred_c0[:, -1]).detach().item()
                cace_block = cace_score(s_pred_c1_block[:, -1], s_pred_c0_block[:, -1]).detach().item()
                label_accuracy_perturb = accuracy_score(s_test[:, -1], s_pred_perturb[:, -1] > 0.5)
                label_accuracy_perturb_block = accuracy_score(s_test[:, -1], s_pred_perturb_block[:, -1] > 0.5)

                # save results
                results.append([f'{dataset}_dataset', model_name, seed, s_accuracy, label_accuracy_perturb, label_accuracy_perturb_block, cace, cace_block])
                metric_cols = ['label_accuracy', 'label_accuracy_perturb', 'label_accuracy_perturb_block', 'cace', 'cace_block']
                df = pd.DataFrame(results, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'results_raw.csv'), index=False)
                
                df = df.drop(columns=['dataset'])
                mean_scaled = df.groupby('model').agg(lambda x: np.mean(x)) * 100
                variance_scaled = df.groupby('model').agg(lambda x: np.std(x)) * 100
                formatted_dfs = []
                for model in mean_scaled.index:
                    formatted_dfs.append(pd.DataFrame({col: f"${mean_scaled.loc[model, col]:.2f} \pm {variance_scaled.loc[model, col]:.2f}$" for col in metric_cols}, index=[model]))
                formatted_df = pd.concat(formatted_dfs)
                print(formatted_df.to_string(index=True, float_format='%.2f'))
                formatted_df.to_csv(os.path.join(results_dir, 'intv_results_table.csv'))
                
                interventions.append([f'{dataset}_dataset', model_name, seed, acc_int])
                metric_cols = ['acc_int']
                df = pd.DataFrame(interventions, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'interventions.csv'), index=False)

                interventions_c.append([f'{dataset}_dataset', model_name, seed, acc_int_only_concept])
                metric_cols = ['acc_int_only_concept']
                df = pd.DataFrame(interventions_c, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'interventions_c.csv'), index=False)

                interventions_y.append([f'{dataset}_dataset', model_name, seed, acc_int_only_label])
                metric_cols = ['acc_int_only_label']
                df = pd.DataFrame(interventions_y, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'interventions_y.csv'), index=False)

                interventions_abs.append([f'{dataset}_dataset', model_name, seed, acc_int_abs])
                metric_cols = ['acc_int_abs']
                df = pd.DataFrame(interventions_abs, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'interventions_abs.csv'), index=False)

                pns_list.append([f'{dataset}_dataset', model_name, seed, PNS])
                metric_cols = ['PNS']
                df = pd.DataFrame(pns_list, columns=['dataset', 'model', 'seed'] + metric_cols)
                df.to_csv(os.path.join(results_dir, 'pns.csv'), index=False)