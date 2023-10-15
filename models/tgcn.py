import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import TemporalGraphConv, global_mean_pool

# Define the number of nodes, input features, and output classes based on your data
num_nodes = 9  # Number of nodes (joints)
num_input_features = (
    27  # Number of input features (joint positions, velocities, efforts)
)
num_output_classes = 3  # Number of output classes (adjust as needed)


# Define a simple TGN model
class TGNModel(nn.Module):
    def __init__(self, num_nodes, num_input_features, num_output_classes):
        super(TGNModel, self).__init__()
        self.conv1 = TemporalGraphConv(
            num_input_features, 64
        )  # Temporal Graph Convolution Layer
        self.conv2 = TemporalGraphConv(
            64, 64
        )  # Another Temporal Graph Convolution Layer
        self.fc = nn.Linear(
            64, num_output_classes
        )  # Fully connected layer for classification

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = global_mean_pool(x, data.batch)  # Global pooling over time steps
        x = self.fc(x)
        return x


# Sample data (you should load your dataset here)
# For simplicity, we create a random sample. Replace this with your data loading logic.
num_samples = 1000
num_time_steps = 10
data_list = []

for _ in range(num_samples):
    x = torch.randn(num_time_steps, num_nodes, num_input_features)
    edge_index = torch.tensor(
        [(0, 1, 1, 2, 2, 3), (1, 0, 2, 1, 3, 2)], dtype=torch.long
    )  # Example edge index
    edge_attr = torch.randn(6, num_time_steps)  # Example edge attributes
    batch = torch.tensor(
        [0] * num_time_steps, dtype=torch.long
    )  # Batch index (assuming a single graph)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
    data_list.append(data)

# Create a DataLoader to batch and iterate through the data
batch_size = 64
dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

# Initialize and train the TGN model
model = TGNModel(num_nodes, num_input_features, num_output_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop (customize this according to your dataset)
num_epochs = 10
for epoch in range(num_epochs):
    for batch_data in dataloader:
        optimizer.zero_grad()
        output = model(batch_data)
        labels = torch.randint(num_output_classes, (batch_size,))
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Print loss for monitoring training progress
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

# After training, you can use the model for inference or evaluation.
# For classification, you would typically apply a softmax function to the model's output.
