from sklearn.model_selection import train_test_split
import torch

from input_pipeline.process_bags import process_bag_files, create_csv_files
import config as cfg
from input_pipeline.graph_dataloader import Temporal_Graph_Dataset, Balanced_Dataset
from input_pipeline.balance_dataset import Data_Balancer


# bagfiles_path = "bagdirs/rosbags-spike-2/"
# csv_dir_path = "Data/csv-spike/"

# print(process_bag_files(bagfiles_path))
# print(create_csv_files(bagfiles_path, csv_dir_path))

win_size_in = [60]
win_size_out = [15]
win_shift = [1, 3, 5]


for wi in win_size_in:
    for wo in win_size_out:
        for ws in win_shift:
            print(f"win_size_in: {wi}, win_size_out: {wo}, win_shift: {ws}")
            dataset_path = f"Data/processed_bal/dataset-{wi}-{wo}-{ws}.pt"
            dataset_configs = {
                "WIN_SIZE_IN": wi,
                "WIN_SIZE_OUT": wo,
                "WIN_SHIFT": ws,
            }
            dataset = Temporal_Graph_Dataset(dataset_configs)
            balancer = Data_Balancer(dataset, save=True, save_path=dataset_path)
            dataset = balancer.random_undersampling(type="max")
            print(f"balanced samples: {len(dataset)}")
            # clear the memory
            del dataset

# bal_dataloader = Balanced_Dataset(cfg.dataset_path)
# dataset, class_weights, class_samples = bal_dataloader.load_balanced_dataset()
# print(f"balanced samples: {len(dataset)}")
# print(f"class samples: {class_samples}")
# print(f"class weights: {class_weights}")

# # load the dataset
# dataset_configs = {
#     "WIN_SIZE_IN": cfg.window_size_in,
#     "WIN_SIZE_OUT": cfg.window_size_out,
#     "WIN_SHIFT": cfg.window_shift,
# }
# dataset = Temporal_Graph_Dataset(dataset_configs)
# print(f"Imbalanced samples: {len(dataset)}")

# # balance the train dataset
# balancer = Data_Balancer(dataset, save=False, save_path=cfg.dataset_path)
# dataset = balancer.random_undersampling(type=cfg.bal_type)
# class_samples = balancer.check_balancer()
# class_weights = torch.tensor(balancer.get_class_weights())
# print(f"class samples: {class_samples}")
# print(f"class weights: {class_weights}")
# print(f"balanced samples: {len(dataset)}")
