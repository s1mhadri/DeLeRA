import torch


#################### Parameters not to change ####################

# device to run training on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjacency matrix of the graph with 9 joints
# adj_matrix = torch.tensor(
#     [
#         [0, 1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 1, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     ],
#     dtype=torch.long,
# )

adj_matrix = torch.tensor(
    [
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
    ],
    dtype=torch.long,
)

# Number of nodes in the graph (number of joints in the robot)
num_nodes = 9
# Number of features in the input
# joint_positions, joint_velocities, joint_accelerations, action_states, planner_flag
num_features = 5
# Number of fault classes
num_classes = 7

mean_std_path = "Data/normal_params/mean_std.pt"

#################### Optional Parameters to change ####################
start_epoch = 0

model_name = "STGATLSTM"  # STGCNLSTM, STGCN4LSTM, LSTM, STGATLSTM, RNN

window_size_in = 75
window_size_out = 10
window_shift = 1

batch_size = 64
max_epochs = 50
learning_rate = 0.001

hidden_dim_1 = 128
hidden_dim_2 = 64

num_layers = 2
num_heads = 8
dropout_rate = 0

################ Necessary Parameters to check before running the code ################

project_root = "/home/simha/DeLeRA/"

set_type = "all"
bal_type = "max"
run_num = 0

csv_dir = f"Data/csv-{set_type}/"
dataset_path = (
    f"Data/processed_bal/dataset-{window_size_in}-{window_size_out}-{window_shift}.pt"
)

model_path = f"runs/run-{model_name}-{set_type}-{run_num}/model-top"

loss_graph_save_path = f"Images/loss_graphs/run-{model_name}-{run_num}.png"
cm_save_path = f"Images/confusion_matrices/run-{model_name}-{run_num+1}.png"

"""
train_flag: default: 'fresh'
    'fresh' - train from start,
    'continue'- continue training from a saved checkpoint and start_epoch,
    'eval' - evaluate the model
    'none' - do nothing. Used for creating the dataset
"""
train_flag = "eval"
