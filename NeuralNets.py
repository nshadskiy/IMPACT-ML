import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
from helper.tca import TaylorAnalysis
from models.Models import NNModel


log = logging.getLogger("training")


class EarlyStopping:
    def __init__(self, patience: int = 10, value: float = 0, delta: float = 0):
        """
        Stops the training if the validation loss does not improve based on given critera.

        Args:
            patience: How many epochs to wait after the last time the validation loss improved. (default: 10)
            value: Threshold for stopping if validation and train loss difference exceeds this value. (default: 0)
            delta: Minimal change in the monitored quantity to qualify as an improvement. (default: 0)
        """
        self.patience = patience
        self.value = value
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.min_epoch = 50

    def __call__(self, val_loss, train_loss, epoch):
        val_score = val_loss
        train_score = train_loss

        if self.best_score is None:
            self.best_score = val_score
        # check if validation loss is not improving or the diffrence between train and validation loss is changing too much
        elif self.min_epoch < epoch and (
            val_score > (self.best_score + self.delta)
            or abs(val_score - train_score) / val_score > self.value
        ):
            self.counter += 1
            log.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


class Network:
    def __init__(self, model, data, config, device_to_run, save_path):
        """
        Neural network class with the relevant methods for training and evaluation.
        """
        self.save_path = save_path
        self.device = device_to_run
        self.data = data
        self.config = config
        self.do_tca = self.config["tca"]
        if self.do_tca:
            self.model = TaylorAnalysis(model.to(self.device))
            self.model.setup_tc_checkpoints(
                number_of_variables_in_data=len(self.data.features),
                considered_variables_idx=range(len(self.data.features)),
                variable_names=self.data.features,
                derivation_order=self.config["tca_order"],
                eval_nodes=list(range(len(self.data.classes))) + ["all"],
                eval_only_max_node=False,
            )
        else:
            self.model = model.to(self.device)

        self.best_model_dict = self.model.state_dict()

    def weightedBCELoss(self, y_pred, y_true, weights=None):
        if weights is not None:
            loss_func = nn.BCELoss(weight=weights, reduction="none")
        else:
            loss_func = nn.BCELoss(reduction="none")

        return loss_func(y_pred, y_true)

    def weightedNLLLoss(self, y_pred, y_true, weights=None):
        loss_func = nn.NLLLoss(reduction="none")
        y_pred = F.log_softmax(y_pred, dim=1)

        if weights is not None:
            return loss_func(y_pred, y_true) * weights
        else:
            return loss_func(y_pred, y_true)

    def train(self, epochs=10):
        self.epochs = epochs
        self.learning_rate = self.config["learning_rate"]
        self.L2_lambda = self.config["L2"]
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.L2_lambda
        )

        self.train_loss_log = []
        self.val_loss_log = []
        self.batch_size = self.config["batch_size"]

        # transformation of pandas data frames to torch tensors and sending it to a device (e.g. gpu)
        self.x_train = torch.tensor(self.data.df_train.tolist()).to(self.device)
        self.x_train_weights = torch.tensor(self.data.df_train_weights.tolist()).to(
            self.device
        )
        self.y_train = torch.tensor(self.data.df_train_labels.tolist()).to(self.device)

        self.x_val = torch.tensor(self.data.df_val.tolist()).to(self.device)
        self.x_val_weights = torch.tensor(self.data.df_val_weights.tolist()).to(
            self.device
        )
        self.y_val = torch.tensor(self.data.df_val_labels.tolist()).to(self.device)

        early_stopping = EarlyStopping(
            patience=self.config["early_stopping_patience"],
            value=self.config["early_stopping_value"],
        )
        best_val_loss = 999999.0
        self.best_epoch = -1.0

        for e in range(self.epochs):
            # set model to training mode
            self.model.train()
            t_loss_log = []
            # loop over mini batches
            for i in range(0, self.data.df_train.shape[0], self.batch_size):
                x_batch = self.x_train[i : i + self.batch_size]
                x_batch_weights = self.x_train_weights[i : i + self.batch_size]
                y_batch = self.y_train[i : i + self.batch_size]

                x_var = Variable(x_batch)
                x_var_weights = Variable(x_batch_weights)
                y_var = Variable(y_batch)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                net_out = self.model(x_var)
                weights = x_var_weights / x_var_weights.mean()
                loss = self.weightedNLLLoss(net_out, y_var, weights)
                loss = loss.mean()

                # backward pass + optimization
                loss.backward()
                self.optimizer.step()

                t_loss_log.append(loss.data.cpu().numpy())

            if self.do_tca:
                self.model.tc_checkpoint(self.x_train, epoch=e)

            self.train_loss_log.append(np.mean(t_loss_log))

            # set model to evaluation mode and evaluate the validation data set
            self.model.eval()
            v_loss_log = []
            # loop over mini batches
            for i in range(0, self.data.df_val.shape[0], self.batch_size):
                x_val_batch = self.x_val[i : i + self.batch_size]
                x_val_batch_weights = self.x_val_weights[i : i + self.batch_size]
                y_val_batch = self.y_val[i : i + self.batch_size]

                x_val_var = Variable(x_val_batch)
                x_val_var_weights = Variable(x_val_batch_weights)
                y_val_var = Variable(y_val_batch)

                with torch.no_grad():
                    net_val_out = self.model(x_val_var)
                    val_weights = x_val_var_weights / x_val_var_weights.mean()
                    val_loss = self.weightedNLLLoss(net_val_out, y_val_var, val_weights)
                    val_loss = val_loss.mean()

                v_loss_log.append(val_loss.data.cpu().numpy())

            self.val_loss_log.append(np.mean(v_loss_log))

            if np.mean(v_loss_log) < best_val_loss:
                best_val_loss = np.mean(v_loss_log)
                self.best_epoch = e + 1
                self.best_model_dict = self.model.state_dict()

            log.info(
                "Epoch: {} - Train. Loss: {:.6f} - Val. Loss: {:.6f}".format(
                    e + 1, np.mean(t_loss_log), np.mean(v_loss_log)
                )
            )

            early_stopping(np.mean(v_loss_log), np.mean(t_loss_log), e)

            if early_stopping.early_stop:
                log.info("Early stopping")
                break

    def predict(self):
        self.x_test = torch.tensor(self.data.df_test.tolist()).to(self.device)
        self.y_test = torch.tensor(self.data.df_test_labels.tolist()).to(self.device)

        xT_var = Variable(self.x_test)

        if self.do_tca:
            if not os.path.exists(self.save_path + "/tca"):
                os.makedirs(self.save_path + "/tca")

            self.model.plot_taylor_coefficients(
                self.x_test,
                considered_variables_idx=range(len(self.data.features)),
                variable_names=self.data.features,
                derivation_order=self.config["tca_order"],
                path=[
                    self.save_path + "/tca/coefficients.pdf",
                    self.save_path + "/tca/coefficients.png",
                ],
                eval_nodes=list(range(len(self.data.classes))) + ["all"],
                eval_only_max_node=False,
            )
            self.model.plot_checkpoints(
                path=[
                    self.save_path + "/tca/tc_training.pdf",
                    self.save_path + "/tca/tc_training.png",
                ]
            )

        self.best_model = NNModel(
            n_input_features=len(self.data.features),
            n_output_nodes=len(self.data.classes),
            hidden_layer=self.config["hidden_layers"],
            dropout_p=self.config["dropout_p"],
        )
        self.best_model.load_state_dict(self.best_model_dict)
        self.best_model.to(self.device)
        self.best_model.eval()

        with torch.no_grad():
            self.prediction = self.best_model(xT_var)
            self.prediction = F.softmax(self.prediction, dim=1)
        self.pred_weights = self.data.df_test_weights

        self.roc_auc_scores = roc_auc_score(
            self.y_test.cpu().numpy(),
            self.prediction.cpu().numpy(),
            average=None,
            multi_class="ovr",
        )
        log.info("-" * 50)
        for cl in self.data.label_dict:
            log.info(
                f"{cl} ROC-AUC score: {self.roc_auc_scores[self.data.label_dict[cl]]}"
            )

    def predict_for_mass_points(self):
        self.x_test = dict()
        self.y_test = dict()
        self.prediction = dict()
        self.pred_weights = dict()
        self.roc_auc_scores = dict()

        signal_labels = list()
        for sig in self.data.signal:
            signal_labels.append(self.data.label_dict[sig])

        for comb in self.data.mass_combinations:
            idx_massX = self.data.mass_indizes["massX"][comb["massX"]]
            idx_massY = self.data.mass_indizes["massY"][comb["massY"]]

            if self.config["one_hot_parametrization"]:
                # remove all signal events with wrong mass points
                mask = (self.data.df_test_labels.isin(signal_labels)) & ~(
                    (self.data.df_test[f"massX_{idx_massX}"].astype(int))
                    & (self.data.df_test[f"massY_{idx_massY}"].astype(int))
                )
                log.info(
                    f"event number of not massX {comb['massX']}, massY {comb['massY']}: {sum(mask)}"
                )
                mask = ~mask
                df_test = self.data.df_test[mask].copy(deep=True)
                df_test_labels = self.data.df_test_labels[mask].copy(deep=True)
                # change all mass points to the same values
                for param in self.data.param_features:
                    if param == f"massX_{idx_massX}":
                        df_test[param] = 1
                    elif param == f"massY_{idx_massY}":
                        df_test[param] = 1
                    else:
                        df_test[param] = 0
            else:
                # remove all signal events with wrong mass points
                mask = ~(
                    (self.data.df_test_labels.isin(signal_labels))
                    & ~(
                        (self.data.df_test["massX"].isin([idx_massX]))
                        & (self.data.df_test["massY"].isin([idx_massY]))
                    )
                )
                df_test = self.data.df_test[mask].copy(deep=True)
                df_test_labels = self.data.df_test_labels[mask].copy(deep=True)
                # change all mass points to the same values
                for param in self.data.param_features:
                    df_test[param] = self.data.mass_indizes[param][comb[param]]

            self.x_test[f"massX_{comb['massX']}_massY_{comb['massY']}"] = torch.tensor(
                df_test.values.tolist()
            ).to(self.device)
            self.y_test[f"massX_{comb['massX']}_massY_{comb['massY']}"] = torch.tensor(
                df_test_labels.values.tolist()
            ).to(self.device)

            xT_var = Variable(
                self.x_test[f"massX_{comb['massX']}_massY_{comb['massY']}"]
            )

            if self.do_tca:
                if not os.path.exists(self.save_path + "/tca"):
                    os.makedirs(self.save_path + "/tca")

                self.model.plot_taylor_coefficients(
                    self.x_test[f"massX_{comb['massX']}_massY_{comb['massY']}"],
                    considered_variables_idx=range(len(self.data.features)),
                    variable_names=self.data.features,
                    derivation_order=self.config["tca_order"],
                    path=[
                        self.save_path + "/tca/coefficients.pdf",
                        self.save_path + "/tca/coefficients.png",
                    ],
                    eval_nodes=list(range(len(self.data.classes))) + ["all"],
                    eval_only_max_node=False,
                )
                self.model.plot_checkpoints(
                    path=[
                        self.save_path + "/tca/tc_training.pdf",
                        self.save_path + "/tca/tc_training.png",
                    ]
                )

            self.best_model = NNModel(
                n_input_features=len(self.data.features + self.data.param_features),
                n_output_nodes=len(self.data.classes),
                hidden_layer=self.config["hidden_layers"],
                dropout_p=self.config["dropout_p"],
            )
            self.best_model.load_state_dict(self.best_model_dict)
            self.best_model.to(self.device)
            self.best_model.eval()

            with torch.no_grad():
                self.prediction[
                    f"massX_{comb['massX']}_massY_{comb['massY']}"
                ] = F.softmax(self.best_model(xT_var), dim=1)
            self.pred_weights[
                f"massX_{comb['massX']}_massY_{comb['massY']}"
            ] = self.data.df_test_weights[mask].values

            self.roc_auc_scores[
                f"massX_{comb['massX']}_massY_{comb['massY']}"
            ] = roc_auc_score(
                self.y_test[f"massX_{comb['massX']}_massY_{comb['massY']}"]
                .cpu()
                .numpy(),
                self.prediction[f"massX_{comb['massX']}_massY_{comb['massY']}"]
                .cpu()
                .numpy(),
                average=None,
                multi_class="ovr",
            )
            log.info("-" * 50)
            log.info(f"ROC-AUC scores for massX={comb['massX']}, massY={comb['massY']}")
            for cl in self.data.label_dict:
                rocs = self.roc_auc_scores[
                    f"massX_{comb['massX']}_massY_{comb['massY']}"
                ][self.data.label_dict[cl]]
                log.info(f"{cl} ROC-AUC score: {rocs}")
