from pathlib import Path

import wandb
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from input_pipeline.graph_dataloader import Temporal_Graph_Dataset, Balanced_Dataset
from input_pipeline.balance_dataset import Data_Balancer
from models.graph_models import STGCNLSTM, STGCN4LSTM
from trainer import Trainer, create_train_loss_graph
from evaluate import Evaluate_Model


def train_func():
    wandb.init()
    wbconfigs = wandb.config

    model_path_dir = cfg.model_path.rsplit("/", 1)[0]
    Path(model_path_dir).mkdir(parents=True, exist_ok=True)

    dataset_path = f"Data/processed_bal/dataset-{wbconfigs.WIN_SIZE_IN}-{wbconfigs.WIN_SIZE_OUT}-{wbconfigs.WIN_SHIFT}.pt"
    bal_dataloader = Balanced_Dataset(dataset_path)
    dataset, class_weights, class_samples = bal_dataloader.load_balanced_dataset()

    # split the dataset into train, val, and test
    train, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)

    # create the dataloaders
    batch_size = wbconfigs.BATCH_SIZE
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    # Instantiate the GCN model
    model_configs = {
        "NUM_NODES": cfg.num_nodes,
        "NUM_FEATURES": cfg.num_features,
        "NUM_CLASSES": cfg.num_classes,
        "WIN_SIZE_IN": wbconfigs.WIN_SIZE_IN,
        "WIN_SIZE_OUT": wbconfigs.WIN_SIZE_OUT,
        "HIDDEN_DIM_1": wbconfigs.HIDDEN_DIM_1,
        "HIDDEN_DIM_2": wbconfigs.HIDDEN_DIM_2,
        "NUM_LAYERS": wbconfigs.NUM_LAYERS,
        "DROPOUT_RATE": wbconfigs.DROPOUT_RATE,
    }
    model = STGCN4LSTM(model_configs)
    # Define loss function and optimizer
    class_weights = torch.tensor(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(cfg.device))
    optimizer = optim.Adam(model.parameters(), lr=wbconfigs.LEARNING_RATE)

    # Train the model from scratch
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        cfg.device,
        cfg.model_path,
    )
    trained_model, (t_losses, v_losses) = trainer.train_model(
        max_epochs=wbconfigs.MAX_EPOCHS,
        log_flag=True,
    )
    # Evaluate the model
    evaluator = Evaluate_Model(trained_model, val_loader, cfg.device)
    predicts, targets = evaluator.evaluate()
    precision, recall, f1_score = evaluator.check_metrics(predicts, targets)
    # Log the metrics
    wandb.log({"precision": precision, "recall": recall, "f1_score": f1_score})


# Configuration for the hyperparameter tuning.
sweep_config = {
    "name": "DeLeRA-hptune",
    "parameters": {
        "WIN_SIZE_IN": {"values": [25, 35, 50, 60, 75, 90]},
        "WIN_SIZE_OUT": {"values": [5, 10, 15]},
        "WIN_SHIFT": {"values": [1, 3, 5]},
        "BATCH_SIZE": {"values": [16, 32, 64, 128]},
        "MAX_EPOCHS": {
            "distribution": "int_uniform",
            "min": 10,
            "max": 75,
        },
        "LEARNING_RATE": {
            "distribution": "log_uniform_values",
            "min": 0.0001,
            "max": 0.1,
        },
        "HIDDEN_DIM_1": {"values": [16, 32, 64, 128]},
        "HIDDEN_DIM_2": {"values": [16, 32, 64, 128]},
        "NUM_LAYERS": {"values": [1, 2, 3, 4]},
        "DROPOUT_RATE": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.4,
        },
    },
    "metric": {"goal": "maximize", "name": "f1_score"},
    "method": "bayes",
}

# Create a sweep
sweep_id = wandb.sweep(sweep_config, project="DeLeRA")

# Perfrom the sweep
wandb.agent(sweep_id, function=train_func, count=100)
