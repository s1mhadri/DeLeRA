from pathlib import Path

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


def load_best_model(model_path):
    # load the model from best saved checkpoint
    load_path = f"{model_path}-1.pt"
    chkpt = torch.load(load_path)
    trained_model = chkpt["model"]
    trained_model.load_state_dict(chkpt["model_sd"])
    return trained_model


def get_model(model_name, model_configs):
    if model_name == "STGCNLSTM":
        model = STGCNLSTM(model_configs)
    elif model_name == "STGCN4LSTM":
        model = STGCN4LSTM(model_configs)
    else:
        raise f"Model {model_name} not implemented"
    return model


def run_main():
    # load all the parameters from config.py
    device = cfg.device
    batch_size = cfg.batch_size
    model_path = cfg.model_path
    train_flag = cfg.train_flag

    # check if the model path exists
    model_path_dir = model_path.rsplit("/", 1)[0]
    Path(model_path_dir).mkdir(parents=True, exist_ok=True)

    loss_graph_path = cfg.loss_graph_save_path
    save_path = cfg.cm_save_path

    # check the device to run the training on
    print(f"Running on: {device}")

    # load the dataset
    # dataset_configs = {
    #     "WIN_SIZE_IN": cfg.window_size_in,
    #     "WIN_SIZE_OUT": cfg.window_size_out,
    #     "WIN_SHIFT": cfg.window_shift,
    # }
    # dataset = Temporal_Graph_Dataset(dataset_configs, load=True)
    # print(f"Imbalanced samples: {len(dataset)}")

    bal_dataloader = Balanced_Dataset(cfg.dataset_path)
    dataset, class_weights, class_samples = bal_dataloader.load_balanced_dataset()

    # balance the train dataset
    # balancer = Data_Balancer(dataset, save=False)
    # dataset = balancer.random_undersampling(type="max")
    # class_samples = balancer.check_balancer()
    # class_weights = torch.tensor(balancer.get_class_weights())
    print(f"class samples: {class_samples}")
    print(f"class weights: {class_weights}")

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
    model_configs = {
        "NUM_NODES": cfg.num_nodes,
        "NUM_FEATURES": cfg.num_features,
        "NUM_CLASSES": cfg.num_classes,
        "WIN_SIZE_IN": cfg.window_size_in,
        "WIN_SIZE_OUT": cfg.window_size_out,
        "HIDDEN_DIM_1": cfg.hidden_dim_1,
        "HIDDEN_DIM_2": cfg.hidden_dim_2,
        "NUM_LAYERS": cfg.num_layers,
        "DROPOUT_RATE": cfg.dropout_rate,
    }
    model = get_model(cfg.model_name, model_configs)
    # print(model)
    # Define loss function and optimizer
    class_weights = torch.tensor(class_weights)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    if train_flag == "fresh":
        # Train the model from scratch
        trainer = Trainer(
            model, optimizer, criterion, train_loader, val_loader, device, model_path
        )
        trained_model, (t_losses, v_losses) = trainer.train_model(
            max_epochs=cfg.max_epochs
        )
        # create the train loss graph
        create_train_loss_graph(
            t_losses, v_losses, save_flag=True, save_path=loss_graph_path
        )
        # load the best model from the saved checkpoints
        trained_model = load_best_model(model_path)
    elif train_flag == "continue":
        # train the model from a saved checkpoint
        trainer = Trainer(
            model, optimizer, criterion, train_loader, val_loader, device, model_path
        )
        trained_model, (t_losses, v_losses) = trainer.train_model(
            cfg.max_epochs, cfg.start_epoch, train
        )
        # create the train loss graph
        create_train_loss_graph(t_losses, v_losses, save_flag=False)
        # load the best model from the saved checkpoints
        trained_model = load_best_model(model_path)
    elif train_flag == "eval":
        # load the best model from the saved checkpoints
        trained_model = load_best_model(model_path)
    else:
        return

    # Evaluate the model
    evaluator = Evaluate_Model(trained_model, test_loader, device)
    predicts, targets = evaluator.evaluate()
    class_report = evaluator.check_classification_report(predicts, targets)
    print(class_report)
    evaluator.create_confusion_matrix(
        predicts, targets, save_flag=True, save_path=save_path
    )


if __name__ == "__main__":
    run_main()
