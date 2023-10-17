import numpy as np
from tqdm import tqdm
import torch


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
        self.avg_len = 0

    def balance(self):
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

        self.avg_len = int(
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
        bal_data = self.data0[:self.avg_len]
        bal_data.extend(self.data1)
        bal_data.extend(self.data2)
        bal_data.extend(self.data3)
        bal_data.extend(self.data4)
        bal_data.extend(self.data5)
        bal_data.extend(self.data6)

        if self.save:
            torch.save(bal_data, self.save_path)

        return bal_data

    def check_balancer(self):
        return (
            [
                # len(self.data0),
                self.avg_len,
                len(self.data1),
                len(self.data2),
                len(self.data3),
                len(self.data4),
                len(self.data5),
                len(self.data6),
            ]
        )
    
    def get_class_weights(self):
        num_classes = 7
        class_samples = self.check_balancer()
        total_samples = sum(class_samples)
        class_weights = [total_samples / (num_classes * class_samples[i]) for i in range(num_classes)]
        
        return class_weights