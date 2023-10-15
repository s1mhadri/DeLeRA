import glob

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

import config as cfg


class Temporal_Graph_Dataset(Dataset):
    """
    Dataset class for creating or loading graph data from csv files

    Parameters:
        load: bool, default: True
            If True, load the data from the processed directory else create the
            dataset and save it in the processed directory
    """

    def __init__(self, dataset_path, csv_dir, load=True):
        self.csv_dir = csv_dir
        self.num_nodes = cfg.num_nodes
        self._num_features = cfg.num_features
        self.num_classes = cfg.num_classes
        self.adj_matrix = cfg.adj_matrix
        self.processed_dir = dataset_path
        # check if data_list.pt exists
        if load:
            print("Loading dataset from: ", self.processed_dir)
            self.data_list = torch.load(self.processed_dir)
        else:
            self.create_window()
            self.data_list = torch.load(self.processed_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @property
    def num_features(self):
        return self._num_features

    @property
    def processed_file_names(self) -> str:
        return self.processed_dir

    def create_window(self):
        files = glob.glob(self.csv_dir + "*.csv")
        files.sort()

        win_size_in = cfg.window_size_in
        win_size_out = cfg.window_size_out
        win_shift = cfg.window_shift
        win_size_total = win_size_in + win_size_out

        data_list = []
        for file in tqdm(files, desc="Creating dataset"):
            filedata = torch.tensor(pd.read_csv(file).values, dtype=torch.float)
            for i in range((filedata.shape[0] - win_size_total) // win_shift + 1):
                joint_feats = filedata[
                    i * win_shift : i * win_shift + win_size_in, 3:-2
                ]  # (win_size_in, 27)
                joint_feats = joint_feats.view(-1, 3, 9).permute(
                    0, 2, 1
                )  # (win_size_in, 27) -> (win_size_in, 9, 3)
                other_feats = filedata[
                    i * win_shift : i * win_shift + win_size_in, 1:3
                ]  # (win_size_in, 2)
                other_feats = other_feats.repeat(9, 1).view(
                    -1, 9, 2
                )  # (win_size_in, 2) -> (win_size_in, 9, 2)
                node_feats = torch.cat(
                    (joint_feats, other_feats), dim=-1
                )  # (win_size_in, 9, 5)
                label = torch.unsqueeze(
                    filedata[
                        i * win_shift + win_size_in : i * win_shift + win_size_total, -1
                    ],
                    0,
                )
                # convert label to torch.long
                label = label.long()
                data = Data(
                    x=node_feats, edge_index=self.adj_matrix.nonzero().t(), y=label
                )
                data_list.append(data)

        torch.save(data_list, self.processed_dir)
        print("Dataset created and saved at: ", self.processed_dir)


class Static_Graph_Dataset(Dataset):
    """
    Dataset class for creating or loading graph data from csv files

    Parameters:
        load: bool, default: True
            If True, load the data from the processed directory else create the
            dataset and save it in the processed directory
    """

    def __init__(self, load=True):
        self.csv_dir = cfg.csv_dir
        self.num_nodes = cfg.num_nodes
        self._num_features = cfg.num_features
        self.num_classes = cfg.num_classes
        self.adj_matrix = cfg.adj_matrix
        self.processed_dir = cfg.processed_data_path
        # check if data_list.pt exists
        if load:
            self.data_list = torch.load(self.processed_dir)
        else:
            self.load_data()
            self.data_list = torch.load(self.processed_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @property
    def num_features(self):
        return self._num_features

    @property
    def processed_file_names(self) -> str:
        return self.processed_dir

    def load_data(self):
        files = glob.glob(self.csv_dir + "*.csv")
        files.sort()
        data_list = []
        for file in tqdm(files, desc="Creating dataset"):
            filedata = torch.tensor(pd.read_csv(file).values, dtype=torch.float)
            for i in range(1, filedata.shape[0]):
                joint_feats = filedata[i - 1, 3:-2]
                joint_feats = joint_feats.view(3, 9).t()
                other_feats = filedata[i - 1, 1:3]
                other_feats = other_feats.expand(9, 2)
                node_feats = torch.cat((joint_feats, other_feats), dim=1)  # (9, 5)
                label = torch.unsqueeze(filedata[i, -1], 0)
                # convert label to torch.long
                label = label.long()
                data = Data(
                    x=node_feats, edge_index=self.adj_matrix.nonzero().t(), y=label
                )
                data_list.append(data)

        torch.save(data_list, self.processed_dir)
        print("Dataset created and saved at: ", self.processed_dir)
