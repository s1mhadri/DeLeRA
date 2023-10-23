import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class Evaluate_Model:
    def __init__(self, model, test_loader, device):
        self.device = device
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

    def check_classification_report(self, predicts, targets):
        predicts = np.concatenate(predicts).ravel()
        targets = np.concatenate(targets).ravel()
        labels = [0, 1, 2, 3, 4, 5, 6]
        target_names = [
            "no fe",
            "ctrl fail",
            "crit acc",
            "pick fail",
            "rel fail",
            "collision",
            "thrown",
        ]
        class_report = classification_report(
            targets, predicts, labels=labels, target_names=target_names, zero_division=0
        )
        return class_report

    def create_confusion_matrix(
        self, predicts, targets, save_flag=True, save_path=None
    ):
        predicts = np.concatenate(predicts).ravel()
        targets = np.concatenate(targets).ravel()
        labels = [0, 1, 2, 3, 4, 5, 6]
        cm = confusion_matrix(targets, predicts, labels=labels).T
        plt.figure(figsize=(10, 10))
        # increase font size of labels in the heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20})
        plt.xlabel("True Label", fontsize=20)
        plt.ylabel("Predicted Label", fontsize=20)
        plt.title("Confusion Matrix", fontsize=20)
        if save_flag and save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
