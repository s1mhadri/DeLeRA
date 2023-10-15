import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import config as cfg


class Evaluate_Model:
    def __init__(self, model, test_loader):
        self.device = cfg.device
        self.model = model
        self.test_loader = test_loader
        self.predicts = []
        self.targets = []

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                predicted = torch.argmax(output, dim=-1)
                # _, predicted = torch.max(output, 1)
                self.predicts.append(predicted.cpu().detach().numpy())
                self.targets.append(batch.y.view(-1).cpu().detach().numpy())

        return self.predicts, self.targets

    def check_metrics(self, predicts, targets):
        predicts = np.concatenate(predicts).ravel()
        targets = np.concatenate(targets).ravel()
        precision = precision_score(targets, predicts, average="macro")
        recall = recall_score(targets, predicts, average="macro")
        f1 = f1_score(targets, predicts, average="macro")
        return precision, recall, f1

    def create_confusion_matrix(self, predicts, targets):
        predicts = np.concatenate(predicts).ravel()
        targets = np.concatenate(targets).ravel()
        cm = confusion_matrix(targets, predicts, labels=[0, 1, 2, 3, 4, 5, 6]).T
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("True Label")
        plt.ylabel("Predicted Label")
        plt.show()
