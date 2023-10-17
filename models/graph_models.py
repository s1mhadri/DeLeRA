import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


# Define the ST-GCN with LSTM model
class STGCNLSTM(nn.Module):
    def __init__(self, configs):
        super(STGCNLSTM, self).__init__()
        self.num_nodes = configs["NUM_NODES"]
        self.num_features = configs["NUM_FEATURES"]
        self.num_classes = configs["NUM_CLASSES"]
        self.win_size_in = configs["WIN_SIZE_IN"]
        self.win_size_out = configs["WIN_SIZE_OUT"]
        self.hidden_dim_1 = configs["HIDDEN_DIM_1"]
        self.hidden_dim_2 = configs["HIDDEN_DIM_2"]
        self.num_layers = configs["NUM_LAYERS"]

        # Spatial Graph Convolution Layers
        self.conv1 = GCNConv(self.num_features, self.hidden_dim_1)
        self.conv2 = GCNConv(self.hidden_dim_1, self.num_features)
        # LSTM Layer
        self.lstm = nn.LSTM(
            self.num_nodes * self.num_features,
            self.hidden_dim_1,
            batch_first=True,
            num_layers=self.num_layers,
        )
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.num_classes)

    def forward(self, data):
        """
        B - batch_size
        N - num_nodes
        F - num_features
        WI - win_size_in
        WO - win_size_out
        H1 - hidden_dim_1
        H2 - hidden_dim_2
        C - num_classes
        """
        x, edge_index = data.x, data.edge_index
        # x: Node features (B, WI, N, F)

        x = x.view(-1, self.num_features)  # (B*WI*N, F)
        # Apply spatial graph convolutions
        x = F.relu(self.conv1(x, edge_index))  # (B*WI*N, H1)
        x = F.relu(self.conv2(x, edge_index))  # (B*WI*N, F)

        batch_size = data.num_graphs
        x = x.view(batch_size, self.win_size_in, -1)  # (B, WI, N*F)
        # Apply LSTM
        x, _ = self.lstm(x)  # (B, WI, H1)
        x = x[:, -self.win_size_out :, :]  # (B, WO, H1)

        # x = x[:, -1, :]  # (B, WO, H1)
        # x = x.view(batch_size, self.win_size_out, -1)  # (B, WO, H1)
        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # (B, WO, H2)
        x = self.fc2(x)  # (B, WO, C)
        x = x.view(-1, self.num_classes)  # (B*WO, C)

        return x


# Define the ST-GCN with LSTM model
class STGCN4LSTM(nn.Module):
    def __init__(self, configs):
        super(STGCN4LSTM, self).__init__()

        self.num_nodes = configs["NUM_NODES"]
        self.num_features = configs["NUM_FEATURES"]
        self.num_classes = configs["NUM_CLASSES"]

        self.win_size_in = configs["WIN_SIZE_IN"]
        self.win_size_out = configs["WIN_SIZE_OUT"]
        self.hidden_dim_1 = configs["HIDDEN_DIM_1"]
        self.hidden_dim_2 = configs["HIDDEN_DIM_2"]
        self.num_layers = configs["NUM_LAYERS"]

        # Spatial Graph Convolution Layers
        self.conv1 = GCNConv(self.num_features, self.hidden_dim_1)
        self.conv2 = GCNConv(self.hidden_dim_1, self.hidden_dim_1)
        self.conv3 = GCNConv(self.hidden_dim_1, self.hidden_dim_1)
        self.conv4 = GCNConv(self.hidden_dim_1, self.num_features)
        # LSTM Layer
        self.lstm = nn.LSTM(
            self.num_nodes * self.num_features,
            self.hidden_dim_1,
            batch_first=True,
            num_layers=self.num_layers,
        )
        # Fully Connected Layers
        self.fc1 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc2 = nn.Linear(self.hidden_dim_2, self.num_classes)

    def forward(self, data):
        """
        B - batch_size
        N - num_nodes
        F - num_features
        WI - win_size_in
        WO - win_size_out
        H1 - hidden_dim_1
        H2 - hidden_dim_2
        C - num_classes
        """
        x, edge_index = data.x, data.edge_index
        # x: Node features (B, WI, N, F)

        x = x.view(-1, self.num_features)  # (B*WI*N, F)
        # Apply spatial graph convolutions
        x = F.relu(self.conv1(x, edge_index))  # (B*WI*N, H1)
        x = F.relu(self.conv2(x, edge_index))  # (B*WI*N, H1)
        x = F.relu(self.conv3(x, edge_index))  # (B*WI*N, H1)
        x = F.relu(self.conv4(x, edge_index))  # (B*WI*N, F)

        batch_size = data.num_graphs
        x = x.view(batch_size, self.win_size_in, -1)  # (B, WI, N*F)
        # Apply LSTM
        x, _ = self.lstm(x)  # (B, WI, H1)
        x = x[:, -self.win_size_out :, :]  # (B, WO, H1)

        # x = x[:, -1, :]  # (B, WO, H1)
        # x = x.view(batch_size, self.win_size_out, -1)  # (B, WO, H1)
        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # (B, WO, H2)
        x = self.fc2(x)  # (B, WO, C)
        x = x.view(-1, self.num_classes)  # (B*WO, C)

        return x
