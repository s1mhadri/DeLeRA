from pathlib import Path

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from input_pipeline.graph_dataloader import Temporal_Graph_Dataset, Static_Graph_Dataset
from models.graph_models import GCN, STGCNLSTM
from trainer import Trainer
from evaluate import Evaluate_Model


def run_main():
    # load all the parameters from config.py
    device = cfg.device
    dataset_path = cfg.dataset_path
    csv_dir = cfg.csv_dir
    batch_size=cfg.batch_size
    model_path = cfg.model_path
    train_flag = cfg.train_flag

    # check if the model path exists
    model_path_dir = model_path.rsplit('/', 1)[0]
    Path(model_path_dir).mkdir(parents=True, exist_ok=True)

    # check the device to run the training on
    print(f"Running on: {device}")

    # load the dataset
    load = Path(dataset_path).is_file()
    dataset = Temporal_Graph_Dataset(dataset_path, csv_dir, load)

    # split the dataset into train, val, and test
    train, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train, test_size=0.2, random_state=42)
    print(
        f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}"
    )

    # create the dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Instantiate the GCN model
    model = STGCNLSTM(
        num_nodes=cfg.num_nodes,
        num_features=cfg.num_features,
        num_classes=cfg.num_classes,
        win_size_in=cfg.window_size_in,
        win_size_out=cfg.window_size_out,
    )
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if train_flag == 'fresh':
        # Train the model from scratch
        trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, device, model_path)
        trained_model, _ = trainer.train_model(max_epochs=cfg.max_epochs)
    elif train_flag == 'continue':
        # train the model from a saved checkpoint
        trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, device, model_path)
        trained_model, _ = trainer.train_model(cfg.max_epochs, cfg.start_epoch, train)
    elif train_flag == 'eval':
        # load the model from best saved checkpoint
        load_path = f'{model_path}-1.pt'
        chkpt = torch.load(load_path)
        trained_model = chkpt['model']
        trained_model.load_state_dict(chkpt['model_sd'])

    # Evaluate the model
    evaluator = Evaluate_Model(trained_model, test_loader)
    predicts, targets = evaluator.evaluate()
    precision, recall, f1 = evaluator.check_metrics(predicts, targets)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    evaluator.create_confusion_matrix(predicts, targets)
    

if __name__ == "__main__":
    run_main()
