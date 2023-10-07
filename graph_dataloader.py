import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

import config as cfg


class Graph_Dataloader():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def create_dataloader(self, dataset):
        data_list = []
        for _ in range(dataset.shape[0]):
            x = torch.randn(cfg.num_nodes, cfg.num_features)
            y = torch.randint(0, cfg.num_classes, (1,))
            data = Data(x=x, edge_index=cfg.adj_matrix.nonzero().t(), y=y)
            data_list.append(data)

        # Split data into training and testing sets
        train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
