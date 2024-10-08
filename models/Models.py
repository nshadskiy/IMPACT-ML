from typing import List
import torch.nn as nn
import torch.nn.functional as F

# import torch.distributions as dist
from models.LinearVariational import LinearVariational
from dataclasses import dataclass


class NNModel(nn.Module):
    def __init__(
        self,
        n_input_features: int,
        n_output_nodes: int,
        hidden_layer: List[int],
        dropout_p: float,
    ):
        super(NNModel, self).__init__()
        self.input_layer_size = n_input_features
        self.output_layer_size = n_output_nodes
        self.hidden_layer_sizes = hidden_layer
        self.dropout_p = dropout_p

        layers = list()
        if len(self.hidden_layer_sizes) == 0:
            layers.append(nn.Linear(self.input_layer_size, self.output_layer_size))
        else:
            for idx, hl in enumerate(self.hidden_layer_sizes):
                if idx == 0:
                    # layers.append(nn.BatchNorm1d(self.input_layer_size, affine=False, track_running_stats=True))
                    layers.append(nn.Linear(self.input_layer_size, hl))
                    layers.append(nn.Tanh())
                    layers.append(nn.Dropout(p=self.dropout_p))
                    # layers.append(
                    #     nn.BatchNorm1d(hl, affine=True, track_running_stats=True)
                    # )
                else:
                    layers.append(nn.Linear(self.hidden_layer_sizes[idx - 1], hl))
                    layers.append(nn.Tanh())
                    layers.append(nn.Dropout(p=self.dropout_p))
                    # layers.append(
                    #     nn.BatchNorm1d(hl, affine=True, track_running_stats=True)
                    # )

            layers.append(
                nn.Linear(self.hidden_layer_sizes[-1], self.output_layer_size)
            )
            layers.append(nn.Softmax(dim=-1))

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        x = self.sequential(x)
        return x


class gModel(nn.Module):
    def __init__(self, n_input_features):
        super(gModel, self).__init__()
        self.input_layer_size = n_input_features
        self.hidden_layer_sizes = [20]

        self.inputLayer = nn.Linear(self.input_layer_size, self.hidden_layer_sizes[0])
        self.outputLayer = nn.Linear(self.hidden_layer_sizes[0], 1)

    def forward(self, x):
        x = F.relu(self.inputLayer(x))
        x = self.outputLayer(x)
        return x


@dataclass
class KL:
    accumulated_kl_div = 0


class VIModel(nn.Module):
    def __init__(self, n_input_features, n_batches):
        super().__init__()
        self.kl_loss = KL

        # self.layers = nn.Sequential(
        #     LinearVariational(in_size, hidden_size, self.kl_loss, n_batches),
        #     nn.ReLU(),
        #     #LinearVariational(hidden_size, hidden_size, self.kl_loss, n_batches),
        #     #nn.ReLU(),
        #     LinearVariational(hidden_size, out_size, self.kl_loss, n_batches)
        # )

        self.input_layer_size = n_input_features
        self.hidden_layer_sizes = [100, 100, 100]

        self.inputLayer = LinearVariational(
            self.input_layer_size, self.hidden_layer_sizes[0], self.kl_loss, n_batches
        )
        self.hiddenLayer1 = LinearVariational(
            self.hidden_layer_sizes[0],
            self.hidden_layer_sizes[1],
            self.kl_loss,
            n_batches,
        )
        self.hiddenLayer2 = LinearVariational(
            self.hidden_layer_sizes[1],
            self.hidden_layer_sizes[2],
            self.kl_loss,
            n_batches,
        )
        # self.hiddenLayer3 = LinearVariational(self.hidden_layer_sizes[2], self.hidden_layer_sizes[3], self.kl_loss, n_batches)
        self.outputLayer = LinearVariational(
            self.hidden_layer_sizes[0], 1, self.kl_loss, n_batches
        )

    @property
    def accumulated_kl_div(self):
        return self.kl_loss.accumulated_kl_div

    def reset_kl_div(self):
        self.kl_loss.accumulated_kl_div = 0

    def forward(self, x):
        x, mu, sigma = self.inputLayer(x)
        x = F.relu(x)
        x, mu, sigma = self.hiddenLayer1(x)
        x = F.relu(x)
        x, mu, sigma = self.hiddenLayer2(x)
        x = F.relu(x)
        # x, mu, sigma = self.hiddenLayer3(x)
        # x = F.relu(x)
        x, mu, sigma = self.outputLayer(x)
        # x = dist.Normal(x[:,0], x[:,1])
        # print(torch.log(1 + torch.exp(x[:,1])))
        # epsi = torch.full(x[:,1].size(),1e-3).to('cuda:0')
        # psigma = torch.log(1 + torch.exp(x[:,1]))

        # x = dist.Normal(x[:,0], torch.where(x[:,1]<=1e-3, epsi, psigma))
        # psigma = torch.log(1 + torch.exp(x[:,1]))
        # eps = torch.randn_like(psigma)

        # return x[:,0] + (eps * psigma), x[:,0], psigma
        return x, mu, sigma
