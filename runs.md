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
Creating dataset: 866/866 [02:23] Data/processed/all-avgbal-50-10-1.pt  
Imbalanced samples: 1008203   
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


## run-2

### configs
window_size_in = 25
window_size_out = 10
window_shift = 1

batch_size = 16
max_epochs = 20
learning_rate = 0.001

hidden_dim_1 = 64
hidden_dim_2 = 32

num_layers = 2

### dataset
Loading dataset from:  Data/processed/all-maxbal-25-10-1.pt  
Imbalanced samples: 61358  
class samples: [23971, 6816, 23971, 7357, 385, 3217, 5134]  
class weights: tensor([ 0.4222,  1.4850,  0.4222,  1.3758, 26.2898,  3.1463,  1.9715])  
Train samples: 43593, Val samples: 10899, Test samples: 13624  

### training
Epoch [1/20], Loss: 0.8587, Val Loss: 0.5994  
Epoch [2/20], Loss: 0.5565, Val Loss: 0.5220  
Epoch [3/20], Loss: 0.4633, Val Loss: 0.4479  
Epoch [4/20], Loss: 0.4168, Val Loss: 0.4335  
Epoch [5/20], Loss: 0.3837, Val Loss: 0.3883  
Epoch [6/20], Loss: 0.3556, Val Loss: 0.3505  
Epoch [7/20], Loss: 0.3398, Val Loss: 0.3466  
Epoch [8/20], Loss: 0.3162, Val Loss: 0.3308  
Epoch [9/20], Loss: 0.3011, Val Loss: 0.3070  
Epoch [10/20], Loss: 0.2903, Val Loss: 0.2880  
Epoch [11/20], Loss: 0.2888, Val Loss: 0.3323  
Epoch [12/20], Loss: 0.2697, Val Loss: 0.2914  
Epoch [13/20], Loss: 0.2618, Val Loss: 0.2676  
Epoch [14/20], Loss: 0.2543, Val Loss: 0.3077  
Epoch [15/20], Loss: 0.2449, Val Loss: 0.2450  
Epoch [16/20], Loss: 0.2373, Val Loss: 0.2557  
Epoch [17/20], Loss: 0.2318, Val Loss: 0.2493  
Epoch [18/20], Loss: 0.2305, Val Loss: 0.2415  
Epoch [19/20], Loss: 0.2114, Val Loss: 0.2277  
Epoch [20/20], Loss: 0.2124, Val Loss: 0.2285  

### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.94    |   0.85   |   0.89   |  75,437 |
|   ctrl fail   |   0.51    |   0.92   |   0.66   |  5,903  |
|    crit acc   |   0.90    |   0.86   |   0.88   |  35,137 |
|   pick fail   |   0.82    |   0.98   |   0.89   |  9,938  |
|    rel fail   |   0.68    |   0.98   |   0.81   |   285   |
|   collision   |   0.74    |   0.98   |   0.84   |  3,935  |
|     thrown    |   0.79    |   0.96   |   0.87   |  5,605  |
|   Accuracy    |           |          |   0.87   | 136,240 |
|   Macro Avg   |   0.77    |   0.93   |   0.83   | 136,240 |
| Weighted Avg  |   0.89    |   0.87   |   0.88   | 136,240 |


## run-3
### configs
window_size_in = 50  
window_size_out = 10  
window_shift = 1  

batch_size = 16  
max_epochs = 20  
learning_rate = 0.001  

hidden_dim_1 = 64  
hidden_dim_2 = 32  

num_layers = 2  

### dataset
Loading dataset from:  Data/processed/all-maxbal-50-10-1.pt  
Imbalanced samples: 61358  
class samples: [23971, 6816, 23971, 7357, 385, 3217, 5134]  
class weights: tensor([ 0.4222,  1.4850,  0.4222,  1.3758, 26.2898,  3.1463,  1.9715])  
Train samples: 43593, Val samples: 10899, Test samples: 13624  

### training
Epoch [1/20], Loss: 0.9012, Val Loss: 0.6634  
Epoch [2/20], Loss: 0.5443, Val Loss: 0.4917  
Epoch [3/20], Loss: 0.4482, Val Loss: 0.4204  
Epoch [4/20], Loss: 0.4069, Val Loss: 0.3669  
Epoch [5/20], Loss: 0.3659, Val Loss: 0.4142  
Epoch [6/20], Loss: 0.3415, Val Loss: 0.3427  
Epoch [7/20], Loss: 0.3223, Val Loss: 0.3099  
Epoch [8/20], Loss: 0.3062, Val Loss: 0.3025  
Epoch [9/20], Loss: 0.2883, Val Loss: 0.3050  
Epoch [10/20], Loss: 0.2773, Val Loss: 0.2852  
Epoch [11/20], Loss: 0.2634, Val Loss: 0.2813  
Epoch [12/20], Loss: 0.2565, Val Loss: 0.2943  
Epoch [13/20], Loss: 0.2504, Val Loss: 0.2607  
Epoch [14/20], Loss: 0.2343, Val Loss: 0.2410  
Epoch [15/20], Loss: 0.2295, Val Loss: 0.2315  
Epoch [16/20], Loss: 0.2183, Val Loss: 0.2446  
Epoch [17/20], Loss: 0.2185, Val Loss: 0.2302  
Epoch [18/20], Loss: 0.2145, Val Loss: 0.2199  
Epoch [19/20], Loss: 0.2003, Val Loss: 0.2113  
Epoch [20/20], Loss: 0.1978, Val Loss: 0.2317  

### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.94    |   0.87   |   0.90   |  75,671 |
|   ctrl fail   |   0.60    |   0.86   |   0.71   |  5,891  |
|    crit acc   |   0.92    |   0.86   |   0.89   |  34,934 |
|   pick fail   |   0.80    |   0.99   |   0.88   |  10,018 |
|    rel fail   |   0.85    |   0.97   |   0.91   |   279   |
|   collision   |   0.77    |   0.98   |   0.86   |  3,694  |
|     thrown    |   0.79    |   0.98   |   0.88   |  5,753  |
|   Accuracy    |           |          |   0.88   | 136,240 |
|   Macro Avg   |   0.81    |   0.93   |   0.86   | 136,240 |
| Weighted Avg  |   0.90    |   0.88   |   0.89   | 136,240 |


## run-4
### configs
window_size_in = 75  
window_size_out = 10  
window_shift = 1  

batch_size = 16  
max_epochs = 20  
learning_rate = 0.001  

hidden_dim_1 = 64  
hidden_dim_2 = 32  

num_layers = 2  

### dataset
Loading dataset from:  Data/processed/all-maxbal-75-10-1.pt  
Imbalanced samples: 61358  
class samples: [23971, 6816, 23971, 7357, 385, 3217, 5134]  
class weights: tensor([ 0.4222,  1.4850,  0.4222,  1.3758, 26.2898,  3.1463,  1.9715])  
Train samples: 43593, Val samples: 10899, Test samples: 13624  

### training
Epoch [1/20], Loss: 0.8888, Val Loss: 0.6351  
Epoch [2/20], Loss: 0.5527, Val Loss: 0.4810  
Epoch [3/20], Loss: 0.4610, Val Loss: 0.4323  
Epoch [4/20], Loss: 0.4115, Val Loss: 0.3992  
Epoch [5/20], Loss: 0.3756, Val Loss: 0.3445  
Epoch [6/20], Loss: 0.3509, Val Loss: 0.3490  
Epoch [7/20], Loss: 0.3199, Val Loss: 0.2956  
Epoch [8/20], Loss: 0.3008, Val Loss: 0.2859  
Epoch [9/20], Loss: 0.2749, Val Loss: 0.2767  
Epoch [10/20], Loss: 0.2790, Val Loss: 0.2516  
Epoch [11/20], Loss: 0.2433, Val Loss: 0.2634  
Epoch [12/20], Loss: 0.2331, Val Loss: 0.2423  
Epoch [13/20], Loss: 0.2186, Val Loss: 0.2096  
Epoch [14/20], Loss: 0.2095, Val Loss: 0.2002  
Epoch [15/20], Loss: 0.2009, Val Loss: 0.2002  
Epoch [16/20], Loss: 0.1967, Val Loss: 0.1891  
Epoch [17/20], Loss: 0.1796, Val Loss: 0.1964  
Epoch [18/20], Loss: 0.1800, Val Loss: 0.1879  
Epoch [19/20], Loss: 0.1679, Val Loss: 0.1746  
Epoch [20/20], Loss: 0.1637, Val Loss: 0.1769  

### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.95    |   0.88   |   0.92   |  76,113 |
|   ctrl fail   |   0.65    |   0.91   |   0.76   |  5,922  |
|    crit acc   |   0.93    |   0.89   |   0.91   |  34,816 |
|   pick fail   |   0.80    |   0.99   |   0.89   |  9,822  |
|    rel fail   |   0.94    |   1.00   |   0.97   |   272   |
|   collision   |   0.81    |   0.99   |   0.89   |  3,614  |
|     thrown    |   0.81    |   0.98   |   0.89   |  5,681  |
|   Accuracy    |           |          |   0.90   | 136,240 |
|   Macro Avg   |   0.84    |   0.95   |   0.89   | 136,240 |
| Weighted Avg  |   0.91    |   0.90   |   0.90   | 136,240 |


## run-5
### configs
model_name = "LSTM"  

window_size_in = 50  
window_size_out = 10  
window_shift = 1  

batch_size = 16  
max_epochs = 10  
learning_rate = 0.001  

hidden_dim_1 = 64  
hidden_dim_2 = 32  

num_layers = 2  
dropout_rate = 0  

### dataset
Loading dataset from:  Data/processed_bal/dataset-50-10-1.pt  
class samples: [21236, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: [0.4127626940774426, 1.4630994110212938, 0.4127626940774426, 1.2088578915223516, 60.4512315270936, 4.6674273543283125, 2.420720400836391]  
Train samples: 39268, Val samples: 9818, Test samples: 12272  

### training
### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.96    |   0.80   |   0.87   |  72,394 |
|   ctrl fail   |   0.46    |   0.90   |   0.61   |  5,143  |
|    crit acc   |   0.84    |   0.86   |   0.85   |  29,386 |
|   pick fail   |   0.80    |   0.98   |   0.88   |  9,631  |
|    rel fail   |   0.20    |   0.99   |   0.33   |    93   |
|   collision   |   0.55    |   0.94   |   0.69   |  2,128  |
|     thrown    |   0.68    |   0.97   |   0.80   |  3,945  |
|   Accuracy    |           |          |   0.84   | 122,720 |
|   Macro Avg   |   0.64    |   0.92   |   0.72   | 122,720 |
| Weighted Avg  |   0.88    |   0.84   |   0.85   | 122,720 |


## run-6
### configs
model_name = "STGCN4LSTM"  

window_size_in = 50  
window_size_out = 10  
window_shift = 1  

batch_size = 16  
max_epochs = 10  
learning_rate = 0.001  

hidden_dim_1 = 64  
hidden_dim_2 = 32  

num_layers = 2  
dropout_rate = 0  

### dataset
Loading dataset from:  Data/processed_bal/dataset-50-10-1.pt  
class samples: [21236, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: [0.4127626940774426, 1.4630994110212938, 0.4127626940774426, 1.2088578915223516, 60.4512315270936, 4.6674273543283125, 2.420720400836391]  
Train samples: 39268, Val samples: 9818, Test samples: 12272  

### training
### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.94    |   0.83   |   0.88   |  72,394 |
|   ctrl fail   |   0.50    |   0.87   |   0.63   |  5,143  |
|    crit acc   |   0.87    |   0.82   |   0.84   |  29,386 |
|   pick fail   |   0.80    |   0.98   |   0.88   |  9,631  |
|    rel fail   |   0.22    |   1.00   |   0.35   |    93   |
|   collision   |   0.52    |   0.95   |   0.67   |  2,128  |
|     thrown    |   0.62    |   0.98   |   0.76   |  3,945  |
|   Accuracy    |           |          |   0.85   | 122,720 |
|   Macro Avg   |   0.64    |   0.92   |   0.72   | 122,720 |
| Weighted Avg  |   0.88    |   0.85   |   0.85   | 122,720 |


## run-7
### configs
model_name = "STGATLSTM"  

window_size_in = 50  
window_size_out = 10  
window_shift = 1  

batch_size = 16  
max_epochs = 10  
learning_rate = 0.001  

hidden_dim_1 = 64  
hidden_dim_2 = 32  

num_layers = 2  
dropout_rate = 0  

### dataset
Loading dataset from:  Data/processed_bal/dataset-50-10-1.pt  
class samples: [21236, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: [0.4127626940774426, 1.4630994110212938, 0.4127626940774426, 1.2088578915223516, 60.4512315270936, 4.6674273543283125, 2.420720400836391]  
Train samples: 39268, Val samples: 9818, Test samples: 12272  

### training
### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.94    |   0.84   |   0.88   |  72,394 |
|   ctrl fail   |   0.52    |   0.87   |   0.65   |  5,143  |
|    crit acc   |   0.87    |   0.83   |   0.85   |  29,386 |
|   pick fail   |   0.81    |   0.97   |   0.89   |  9,631  |
|    rel fail   |   0.29    |   1.00   |   0.45   |    93   |
|   collision   |   0.55    |   0.98   |   0.71   |  2,128  |
|     thrown    |   0.66    |   0.98   |   0.79   |  3,945  |
|   Accuracy    |           |          |   0.85   | 122,720 |
|   Macro Avg   |   0.66    |   0.92   |   0.75   | 122,720 |
| Weighted Avg  |   0.88    |   0.85   |   0.86   | 122,720 |




## run-8
### configs
model_name = "RNN"  

window_size_in = 50  
window_size_out = 10  
window_shift = 1  

batch_size = 16  
max_epochs = 10  
learning_rate = 0.001  

hidden_dim_1 = 64  
hidden_dim_2 = 32  

num_layers = 2  
dropout_rate = 0

### dataset
Loading dataset from:  Data/processed_bal/dataset-50-10-1.pt  
class samples: [21236, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: [0.4127626940774426, 1.4630994110212938, 0.4127626940774426, 1.2088578915223516, 60.4512315270936, 4.6674273543283125, 2.420720400836391]  
Train samples: 39268, Val samples: 9818, Test samples: 12272  

### training
### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.92    |   0.78   |   0.84   |  72,394 |
|   ctrl fail   |   0.44    |   0.83   |   0.58   |  5,143  |
|    crit acc   |   0.82    |   0.72   |   0.77   |  29,386 |
|   pick fail   |   0.75    |   0.97   |   0.85   |  9,631  |
|    rel fail   |   0.17    |   0.96   |   0.29   |    93   |
|   collision   |   0.36    |   0.94   |   0.52   |  2,128  |
|     thrown    |   0.55    |   0.94   |   0.69   |  3,945  |
|   Accuracy    |           |          |   0.79   | 122,720 |
|   Macro Avg   |   0.57    |   0.88   |   0.65   | 122,720 |
| Weighted Avg  |   0.84    |   0.79   |   0.80   | 122,720 |


## run-9
### configs
model_name = "STGATLSTM"  

window_size_in = 75  
window_size_out = 10  
window_shift = 1  

batch_size = 64  
max_epochs = 50  
learning_rate = 0.001  

hidden_dim_1 = 128  
hidden_dim_2 = 64  

num_layers = 2  
dropout_rate = 0  

### dataset
Loading dataset from:  Data/processed_bal/dataset-75-10-1.pt  
class samples: [21236, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: [0.4127626940774426, 1.4630994110212938, 0.4127626940774426, 1.2088578915223516, 60.4512315270936, 4.6674273543283125, 2.420720400836391]  
Train samples: 39268, Val samples: 9818, Test samples: 12272  

### training
### results


## run-10
### configs
model_name = "STGATLSTM"  

window_size_in = 75  
window_size_out = 10  
window_shift = 1  

batch_size = 64  
max_epochs = 50  
learning_rate = 0.001  

hidden_dim_1 = 128  
hidden_dim_2 = 64  

num_layers = 2  
dropout_rate = 0  

### dataset
Loading dataset from:  Data/processed_bal/dataset-75-10-1.pt  
class samples: [21236, 5991, 21236, 7251, 145, 1878, 3621]  
class weights: [0.4127626940774426, 1.4630994110212938, 0.4127626940774426, 1.2088578915223516, 60.4512315270936, 4.6674273543283125, 2.420720400836391]  
Train samples: 39268, Val samples: 9818, Test samples: 12272  

### training
### results
|     Class     | Precision |  Recall  | F1-Score | Support |
|---------------|-----------|----------|----------|---------|
|     no fe     |   0.98    |   0.95   |   0.97   |  72,172 |
|   ctrl fail   |   0.89    |   0.97   |   0.93   |  5,114  |
|    crit acc   |   0.96    |   0.97   |   0.96   |  29,599 |
|   pick fail   |   0.89    |   0.99   |   0.94   |  9,762  |
|    rel fail   |   0.90    |   0.99   |   0.94   |   101   |
|   collision   |   0.85    |   1.00   |   0.92   |  2,034  |
|     thrown    |   0.95    |   0.99   |   0.97   |  3,938  |
|   Accuracy    |           |          |   0.96   | 122,720 |
|   Macro Avg   |   0.92    |   0.98   |   0.95   | 122,720 |
| Weighted Avg  |   0.96    |   0.96   |   0.96   | 122,720 |


## run-11
### configs
### dataset
### training
### results

