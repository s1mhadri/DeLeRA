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

    def __init__(self, configs, load=False, save=False):
        self.win_size_in = configs["WIN_SIZE_IN"]
        self.win_size_out = configs["WIN_SIZE_OUT"]
        self.win_shift = configs["WIN_SHIFT"]
        self.win_size_total = self.win_size_in + self.win_size_out

        self.processed_dir = cfg.dataset_path
        self.csv_dir = cfg.csv_dir

        self._num_nodes = cfg.num_nodes
        self._num_features = cfg.num_features
        self._num_classes = cfg.num_classes
        self.adj_matrix = cfg.adj_matrix

        self.save = save
        self.load = load

        if load:
            print("Loading dataset from: ", self.processed_dir)
            self.data_list = torch.load(self.processed_dir)
        else:
            self.data_list = self.create_window()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def processed_file_names(self) -> str:
        return self.processed_dir

    def create_window(self):
        files = glob.glob(self.csv_dir + "*.csv")
        files.sort()

        data_list = []
        for file in tqdm(files, desc="Creating dataset"):
            filedata = torch.tensor(pd.read_csv(file).values, dtype=torch.float)
            for i in range(
                (filedata.shape[0] - self.win_size_total) // self.win_shift + 1
            ):
                joint_feats = filedata[
                    i * self.win_shift : i * self.win_shift + self.win_size_in, 3:-2
                ]  # (win_size_in, 27)
                joint_feats = joint_feats.view(-1, 3, self._num_nodes).permute(
                    0, 2, 1
                )  # (win_size_in, 27) -> (win_size_in, num_nodes, 3)
                other_feats = filedata[
                    i * self.win_shift : i * self.win_shift + self.win_size_in, 1:3
                ]  # (win_size_in, 2)
                other_feats = other_feats.repeat(self._num_nodes, 1).view(
                    -1, self._num_nodes, 2
                )  # (win_size_in, 2) -> (win_size_in, num_nodes, 2)
                node_feats = torch.cat(
                    (joint_feats, other_feats), dim=-1
                )  # (win_size_in, num_nodes, num_features)
                labels = torch.unsqueeze(
                    filedata[
                        i * self.win_shift
                        + self.win_size_in : i * self.win_shift
                        + self.win_size_total,
                        -1,
                    ],
                    0,
                )
                # convert labels to torch.long
                labels = labels.long()
                data = Data(
                    x=node_feats, edge_index=self.adj_matrix.nonzero().t(), y=labels
                )
                data_list.append(data)

        if self.save:
            torch.save(data_list, self.processed_dir)
            print("Dataset created and saved at: ", self.processed_dir)
        return data_list
