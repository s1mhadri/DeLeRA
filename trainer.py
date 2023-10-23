from pathlib import Path
from math import gcd

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import wandb


class Trainer:
    def __init__(
        self, model, optimizer, criterion, train_loader, val_loader, device, model_path
    ):
        self.device = device
        self.path = model_path
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.top_models = []

    def _train_step(self, batch):
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        preds = self.model(batch)
        targets = batch.y.view(-1)
        loss = self.criterion(preds, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_model(
        self, max_epochs, start_epoch=0, train_flag="fresh", log_flag=False
    ):
        if train_flag == "fresh":
            start_epoch = 0
        elif train_flag == "continue" and self.load_model() == False:
            print("No saved model found. Training from scratch.")
            start_epoch = 0

        for epoch in range(start_epoch, max_epochs):
            epoch_loss = []
            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                loss = self._train_step(batch)
                epoch_loss.append(loss)
                self.losses.append(loss)

            # calculate average loss for the epoch
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            val_loss = self.validate()
            if log_flag:
                wandb.log({"train_loss": epoch_loss, "val_loss": val_loss})
            else:
                print(
                    f"Epoch [{epoch+1}/{max_epochs}], Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            # add condition to save model (only if val loss is less than previous val loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_top3_models(val_loss)

        return self.model, (self.losses, self.val_losses)

    def validate(self):
        epoch_val_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                preds = self.model(batch)
                targets = batch.y.view(-1)
                loss = self.criterion(preds, targets)
                self.val_losses.append(loss.item())
                epoch_val_loss.append(loss.item())

        # calculate average loss for the epoch
        return sum(epoch_val_loss) / len(epoch_val_loss)

    def save_top3_models(self, metric):
        models_to_save = 3
        self.top_models.append((metric, self.model, self.optimizer))
        self.top_models.sort(key=lambda x: x[0])
        self.top_models = self.top_models[:models_to_save]
        for idx, (metric, model, optimizer) in enumerate(self.top_models):
            checkpoint = {
                "model": model,
                "model_sd": model.state_dict(),
                "optimizer": optimizer,
                "optimizer_sd": optimizer.state_dict(),
                "val_loss": metric,
            }
            save_path = f"{self.path}-{idx+1}.pt"
            torch.save(checkpoint, save_path)

    def load_model(self):
        for i in range(3):
            load_path = f"{self.path}-{i+1}.pt"
            if Path(load_path).is_file():
                chkpt = torch.load(load_path)
                model = chkpt["model"]
                optimizer = chkpt["optimizer"]
                metric = chkpt["val_loss"]
                self.top_models.append((metric, model, optimizer))
            else:
                return False

        load_path = f"{self.path}-1.pt"
        chkpt = torch.load(load_path)
        self.model = chkpt["model"]
        self.model.load_state_dict(chkpt["model_sd"])
        self.optimizer = chkpt["optimizer"]
        self.optimizer.load_state_dict(chkpt["optimizer_sd"])
        self.best_val_loss = chkpt["val_loss"]
        return True


def create_train_loss_graph(train_losses, val_losses, save_flag=False, save_path=None):
    total_x = 100
    train_loss_len = len(train_losses)
    val_loss_len = len(val_losses)
    # gcds = gcd(train_loss_len, val_loss_len)
    # if total_x > gcds:
    #     factor_train = train_loss_len // gcds
    #     factor_val = val_loss_len // gcds
    # else:
    factor_train = train_loss_len // total_x
    factor_val = val_loss_len // total_x

    # create moving average of train losses
    train_losses = [
        sum(train_losses[i : i + factor_train]) / factor_train
        for i in range(0, train_loss_len, factor_train)
    ]
    val_losses = [
        sum(val_losses[i : i + factor_val]) / factor_val
        for i in range(0, val_loss_len, factor_val)
    ]

    plt.plot(train_losses[2:], label="Train Loss")
    plt.plot(val_losses[2:], label="Val Loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if save_flag and save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
