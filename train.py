class Train_Model():
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
    
    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch.y)
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
