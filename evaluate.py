import torch


class Evaluate_Model():
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        self.correct = 0
        self.total = 0
    
    def evaluate(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                output = self.model(batch)
                _, predicted = torch.max(output, 1)
                self.total += batch.y.size(0)
                self.correct += (predicted == batch.y).sum().item()
    
    def get_accuracy(self):
        accuracy = self.correct / self.total
        return accuracy
