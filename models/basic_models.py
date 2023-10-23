from torch.nn import functional as F
from torch import nn


class Simple_LSTM(nn.Module):
    def __init__(self, configs):
        super(Simple_LSTM, self).__init__()

        self.num_nodes = configs["NUM_NODES"]
        self.num_features = configs["NUM_FEATURES"]
        self.num_classes = configs["NUM_CLASSES"]

        self.win_size_in = configs["WIN_SIZE_IN"]
        self.win_size_out = configs["WIN_SIZE_OUT"]
        self.hidden_dim_1 = configs["HIDDEN_DIM_1"]
        self.hidden_dim_2 = configs["HIDDEN_DIM_2"]
        self.num_layers = configs["NUM_LAYERS"]
        self.dropout_rate = configs["DROPOUT_RATE"]

        # LSTM Layer
        self.lstm = nn.LSTM(
            self.num_nodes * self.num_features,
            self.hidden_dim_1,
            batch_first=True,
            dropout=self.dropout_rate,
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

        batch_size = data.num_graphs
        x = x.view(batch_size, self.win_size_in, -1)  # (B, WI, N*F)

        # Apply LSTM
        x, _ = self.lstm(x)  # (B, WI, H1)
        x = x[:, -self.win_size_out :, :]  # (B, WO, H1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # (B, WO, H2)
        x = self.fc2(x)  # (B, WO, C)
        x = x.view(-1, self.num_classes)  # (B*WO, C)

        return x


class Simple_RNN(nn.Module):
    def __init__(self, configs):
        super(Simple_RNN, self).__init__()

        self.num_nodes = configs["NUM_NODES"]
        self.num_features = configs["NUM_FEATURES"]
        self.num_classes = configs["NUM_CLASSES"]

        self.win_size_in = configs["WIN_SIZE_IN"]
        self.win_size_out = configs["WIN_SIZE_OUT"]
        self.hidden_dim_1 = configs["HIDDEN_DIM_1"]
        self.hidden_dim_2 = configs["HIDDEN_DIM_2"]
        self.num_layers = configs["NUM_LAYERS"]
        self.dropout_rate = configs["DROPOUT_RATE"]

        # LSTM Layer
        self.rnn = nn.RNN(
            self.num_nodes * self.num_features,
            self.hidden_dim_1,
            batch_first=True,
            dropout=self.dropout_rate,
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

        batch_size = data.num_graphs
        x = x.view(batch_size, self.win_size_in, -1)  # (B, WI, N*F)

        # Apply LSTM
        x, _ = self.rnn(x)  # (B, WI, H1)
        x = x[:, -self.win_size_out :, :]  # (B, WO, H1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))  # (B, WO, H2)
        x = self.fc2(x)  # (B, WO, C)
        x = x.view(-1, self.num_classes)  # (B*WO, C)

        return x
