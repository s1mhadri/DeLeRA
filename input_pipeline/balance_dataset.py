import numpy as np
from tqdm import tqdm
import torch

import random

random.seed(42)


class Data_Balancer:
    def __init__(self, imbal_dataset, save=False, save_path=None):
        self.save = save
        self.save_path = save_path
        self.dataset = imbal_dataset
        self.dataset_len = len(imbal_dataset)
        self.data0 = []
        self.data1 = []
        self.data2 = []
        self.data3 = []
        self.data4 = []
        self.data5 = []
        self.data6 = []
        self.len0 = 0
        self.shuffle_dataset()

    def random_undersampling(self, type="avg"):
        for i in tqdm(range(self.dataset_len), desc="Balancing dataset"):
            if (self.dataset[i].y == 0).all():
                self.data0.append(self.dataset[i])
            if (self.dataset[i].y == 1).any():
                self.data1.append(self.dataset[i])
            if (self.dataset[i].y == 2).any():
                self.data2.append(self.dataset[i])
            if (self.dataset[i].y == 3).any():
                self.data3.append(self.dataset[i])
            if (self.dataset[i].y == 4).any():
                self.data4.append(self.dataset[i])
            if (self.dataset[i].y == 5).any():
                self.data5.append(self.dataset[i])
            if (self.dataset[i].y == 6).any():
                self.data6.append(self.dataset[i])

        if type == "max":
            self.len0 = int(
                np.max(
                    [
                        len(self.data1),
                        len(self.data2),
                        len(self.data3),
                        len(self.data4),
                        len(self.data5),
                        len(self.data6),
                    ]
                )
            )
        else:
            self.len0 = int(
                np.mean(
                    [
                        len(self.data1),
                        len(self.data2),
                        len(self.data3),
                        len(self.data4),
                        len(self.data5),
                        len(self.data6),
                    ]
                )
            )
        bal_data = self.data0[: self.len0]
        bal_data.extend(self.data1)
        bal_data.extend(self.data2)
        bal_data.extend(self.data3)
        bal_data.extend(self.data4)
        bal_data.extend(self.data5)
        bal_data.extend(self.data6)

        if self.save:
            class_samples = self.check_balancer()
            class_weights = self.get_class_weights()
            torch.save((bal_data, class_weights, class_samples), self.save_path)

        return bal_data

    def shuffle_dataset(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        shuffled_data_list = [self.dataset[i] for i in indices]
        self.dataset = shuffled_data_list

    def check_balancer(self):
        return [
            # len(self.data0),
            self.len0,
            len(self.data1),
            len(self.data2),
            len(self.data3),
            len(self.data4),
            len(self.data5),
            len(self.data6),
        ]

    def get_class_weights(self):
        num_classes = 7
        class_samples = self.check_balancer()
        total_samples = sum(class_samples)
        class_weights = [
            total_samples / (num_classes * class_samples[i])
            if class_samples[i] != 0
            else 0
            for i in range(num_classes)
        ]

        return class_weights
