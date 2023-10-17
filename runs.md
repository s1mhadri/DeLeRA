## run-1
### configs
window_size_in = 50
window_size_out = 10
window_shift = 1

batch_size = 16
max_epochs = 5
learning_rate = 0.001

hidden_dim_1 = 64
hidden_dim_2 = 32

num_layers = 2

### dataset
Creating dataset: 866/866 [02:23]  
Imbalanced samples: 1008203  
Balancing dataset: 1008203/1008203 [01:03]  
class samples: [6687, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: tensor([ 1.0000,  1.1162,  0.3149,  0.9222, 46.1172,  3.5607,  1.8467])  
Train samples: 29957, Val samples: 7490, Test samples: 9362  

### training
Epoch [1/5], Loss: 0.9199, Val Loss: 0.6444  
Epoch [2/5], Loss: 0.6224, Val Loss: 0.5619  
Epoch [3/5], Loss: 0.5221, Val Loss: 0.4880  
Epoch [4/5], Loss: 0.4717, Val Loss: 0.4694  
Epoch [5/5], Loss: 0.4393, Val Loss: 0.4401  

### results

|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.74    |   0.85   |   0.79   |  43,770 |
|   ctrl fail   |   0.54    |   0.65   |   0.59   |  5,087  |
|    crit acc   |   0.95    |   0.52   |   0.67   |  29,505 |
|   pick fail   |   0.81    |   0.94   |   0.87   |  9,217  |
|    rel fail   |   0.23    |   0.84   |   0.36   |   100   |
|   collision   |   0.41    |   0.89   |   0.56   |  2,057  |
|     thrown    |   0.64    |   0.89   |   0.75   |  3,884  |
|   Accuracy    |           |          |   0.75   |  93,620 |
|   Macro Avg   |   0.62    |   0.80   |   0.66   |  93,620 |
| Weighted Avg  |   0.79    |   0.75   |   0.74   |  93,620 |
