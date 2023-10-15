import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


# Define the basic GCN model
class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


# Define the ST-GCN with LSTM model
class STGCNLSTM(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, win_size_in, win_size_out):
        super(STGCNLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.win_size_in = win_size_in
        self.win_size_out = win_size_out
        self.num_features = num_features
        self.num_classes = num_classes
        # Spatial Graph Convolution Layers
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_features)
        # LSTM Layer
        self.lstm = nn.LSTM(num_nodes * num_features, 32, batch_first=True)
        # Fully Connected Layers
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x: Node features (batch_size, win_size_in, num_nodes, num_features)

        x = x.view(-1, self.num_features)
        # Apply spatial graph convolutions
        x = F.relu(
            self.conv1(x, edge_index)
        )  # (batch_size * win_size_in * num_nodes, 64)
        x = F.relu(
            self.conv2(x, edge_index)
        )  # (batch_size * win_size_in * num_nodes, num_nodes*num_features)

        batch_size = data.num_graphs
        x = x.view(batch_size, self.win_size_in, -1)
        # Apply LSTM
        x, _ = self.lstm(x)  # (batch_size, win_size_in, 32)

        x = x[:, -self.win_size_out :, :]  # (batch_size, win_size_out, 32)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # (batch_size, win_size_out, 16)
        x = self.fc2(x)  # (batch_size, win_size_out, num_classes)
        x = x.view(-1, self.num_classes)  # (batch_size * win_size_out, num_classes)

        return x
