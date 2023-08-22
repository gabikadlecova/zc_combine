import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse


def to_dataset(nets, y_accuracies, batch_size=32, shuffle=True):
    y_accuracies = (y_accuracies - y_accuracies.mean()) / y_accuracies.std()
    ind_graphs = []

    for oo, ops, graph in nets:
        ops = [[(1 if i == o else 0) for o in enumerate(oo)] for i in ops]
        x_features, adjacency = torch.Tensor(ops), torch.Tensor(nx.adjacency_matrix(graph).todense())
        ind = x_features, dense_to_sparse(adjacency)[0]
        ind_graphs.append(ind)

    data = [Data(x=ind[0], edge_index=ind[1], y=y) for ind, y in zip(ind_graphs, y_accuracies)]
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle)


def get_layer_sizes(n_layers, hidden_size, first_size=None, last_size=None):
    for i in range(n_layers):
        if i == 0 and first_size is not None:
            yield first_size, hidden_size
        elif i == (n_layers - 1) and last_size is not None:
            yield hidden_size, last_size
        else:
            yield hidden_size, hidden_size


def apply_layer_dropout_relu(x, layers, p, training):
    for i, lin in enumerate(layers):
        if i != 0:
            x = F.relu(x)
            x = F.dropout(x, p=p, training=training)
        x = lin(x)
    return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_linear=2):
        super().__init__()

        sizes = get_layer_sizes(n_linear, hidden_dim, first_size=input_dim)
        self.linears = nn.ModuleList([nn.Linear(indim, outdim) for indim, outdim in sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_linear - 1)])

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            if i != 0:
                x = self.batch_norms[i - 1](x)
                x = F.relu(x)
            x = lin(x)
        return x


def _get_MLP(n_in, n_hidden, n_linear, i):
    return MLP(n_in if i == 0 else n_hidden, n_hidden, n_linear=n_linear)


class GINConcat(torch.nn.Module):
    def __init__(self, n_node_features=5, n_hidden=32, n_convs=3, n_linear=2, n_mlp_linear=2, dropout=0.1,
                 n_hidden_linear=32):
        super().__init__()

        # convs are followed by batch norm and ReLU
        self.convs = nn.ModuleList(
            [GINConv(_get_MLP(n_node_features, n_hidden, n_mlp_linear, i)) for i in range(n_convs)]
        )
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_convs)])

        # compute input size of linear layers
        lin_dim = (n_hidden * n_convs + n_node_features)

        lin_sizes = get_layer_sizes(n_linear, n_hidden_linear, first_size=lin_dim, last_size=1)
        self.lins = torch.nn.ModuleList([nn.Linear(indim, outdim) for indim, outdim in lin_sizes])

        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # concat readout - inputs
        h_out = []
        h_out.append(global_add_pool(x, batch))

        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            h_out.append(global_add_pool(x, batch))

        h = torch.cat(h_out, dim=1)  # concat readout

        # linear layers
        h = apply_layer_dropout_relu(h, self.lins, self.dropout, self.training)
        return torch.flatten(h)


def train(model: torch.nn.Module, train_loader, n_epochs=10, optimizer=None, criterion=None, verbose=True):
    optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=3e-4)
    criterion = criterion if criterion is not None else torch.nn.MSELoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    epoch_losses = []
    for e in range(n_epochs):
        model.train()
        batch_losses = []

        for data in train_loader:
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.

            # prediction loss
            loss = criterion(out, data.y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            batch_losses.append(loss.detach().cpu().numpy())
        lr_scheduler.step()

        e_loss = np.mean(batch_losses)
        e_std = np.std(batch_losses)
        if verbose:
            print(f"Epoch {e} loss: mean {e_loss}, std {e_std}")
            epoch_losses.append(e_loss)

    return epoch_losses
