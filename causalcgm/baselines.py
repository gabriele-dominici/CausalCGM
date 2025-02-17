import torch
from abc import abstractmethod
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss

class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        self.bce_log = BCEWithLogitsLoss(reduction="mean")

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        y_preds = self.forward(X)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)
        s = torch.cat([c_true, y_true], dim=1)

        loss = self.bce_log(y_preds.squeeze(), s.float().squeeze())
        task_accuracy = accuracy_score(s.squeeze(), y_preds > 0)
        self.log("train_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)
        s = torch.cat([c_true, y_true], dim=1)
        loss = self.bce_log(y_preds.squeeze(), s.float().squeeze())
        task_accuracy = accuracy_score(s.squeeze(), y_preds > 0)
        self.log("val_concept_accuracy", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(1)
        s = torch.cat([c_true, y_true], dim=1)
        loss = self.bce_log(y_preds.squeeze(), s.float().squeeze())
        task_accuracy = accuracy_score(s.squeeze(), y_preds > 0)
        self.log("test_acc", task_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer



class StandardE2E(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
        )

    def _unpack_input(self, I):
        return I[0], I[1], I[2]

    def forward(self, X, explain=False):
        return self.model(X)

class BaseCEM(LightningModule):
    def __init__(self, input_dim, embedding_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )
        self.loss = torch.nn.BCELoss()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.01)
    

class ConceptEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
            concept_family=None,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(2 * emb_size, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 2 * emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(2 * emb_size, 1)
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)
        self.concept_family = concept_family

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False, return_intervened=False):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        c_pred_tmp_list = []
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)
            if self.concept_family is None:
                c_pred = torch.sigmoid(c_pred)
            # Time to check for interventions
            c_int = self._after_interventions(
                prob=c_pred,
                concept_idx=i,
                intervention_idxs=used_int_idxs,
                c_true=c,
                train=train,
            )
            if return_intervened:
                c_pred_list.append(c_int)
            else:
                c_pred_list.append(c_pred)

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_int + context_neg * (1 - c_int)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)




class CEM(BaseCEM):
    def __init__(self, input_dim, embedding_size, n_concepts, n_classes, ce_size, tp_size):
        super().__init__(input_dim, embedding_size)
        self.concept_embedder = ConceptEmbedding(embedding_size, n_concepts, ce_size)
        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(tp_size, tp_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(tp_size, n_classes),
            torch.nn.Sigmoid()
        )
        self.n_symbols = n_concepts + n_classes

    def forward(self, x, c=None, intervention_idxs=None, train=False):
        emb = self.encoder(x)
        c_emb, c_pred = self.concept_embedder(emb, c=c, intervention_idxs=intervention_idxs, train=train, return_intervened=True)
        y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
        s_pred = torch.cat([c_pred, y_pred.unsqueeze(1)], dim=-1)
        return s_pred

    def training_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train, c_train, train=True)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train, train=False)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss


class CBM(BaseCEM):
    def __init__(self, input_dim, embedding_size, n_concepts, n_tasks):
        super().__init__(input_dim, embedding_size)
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, n_concepts),
            torch.nn.Sigmoid()
        )
        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(embedding_size, n_tasks),
            torch.nn.Sigmoid()
        )
        self.n_symbols = n_concepts + n_tasks  

    def forward(self, x, c=None, intervention_idxs=[], train=False):
        emb = self.encoder(x)
        c_pred = self.concept_predictor(emb)
        for idx in intervention_idxs:
            if idx < c_pred.shape[-1]:
                c_pred[:, idx] = c[:, idx]
        y_pred = self.task_predictor(c_pred)
        s_pred = torch.cat([c_pred, y_pred], dim=-1)
        return s_pred

    def training_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train, train=False)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_train, c_train, y_train = batch
        s_pred = self.forward(x_train)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
        s_train = torch.cat([c_train, y_train], dim=-1)
        # compute loss
        loss = self.loss(s_pred, s_train)
        concept_accuracy = accuracy_score(s_train.cpu(), s_pred.detach().cpu() > 0.5)

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_concept_accuracy', concept_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss