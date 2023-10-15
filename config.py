import torch


#################### Parameters not to change ####################

# device to run training on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjacency matrix of the graph with 9 joints
adj_matrix = torch.tensor(
    [
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
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

window_size_in = 25
window_size_out = 10
window_shift = 1

#################### Optional Parameters to change ####################

batch_size = 16
start_epoch = 0
max_epochs = 10
learning_rate = 0.001

hidden_dim = 64

################ Necessary Parameters to check before running the code ################

project_root = "/home/simha/DeLeRA/"
set_type = "all"
csv_dir = f"Data/csv-{set_type}/"
dataset_path = f"Data/processed/{set_type}-dataset.pt"
model_path = f"runs/STGCNLSTM-{set_type}/model-top"

'''
train_flag: default: 'fresh'
    'fresh' - train from start,
    'continue'- continue training from a saved checkpoint and start_epoch,
    'eval' - evaluate the model
    'none' - do nothing. Used for creating the dataset
'''
train_flag = 'eval'  
