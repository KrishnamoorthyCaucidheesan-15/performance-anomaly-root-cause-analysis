# !rm -r sample_data
# !git clone https://github.com/manigalati/usad
# %cd usad

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils

from utils import get_default_device, plot_history, to_device, Evaluation, adjust_predictions_for_neighbourhood,\
    metric_count_histogram, reconstruction_error_plot, reconstruction_error_plot_for_nine_metrics,\
    plot_multivariate_anomaly
from usad import UsadModel, training, testing_individual_metrics

window_size = 5 #originally 12
BATCH_SIZE = 7919
N_EPOCHS = 500  #originally 100
hidden_size = 10

# !nvidia-smi -L
device = get_default_device()

# Read data
normal = pd.read_csv("Datasets/choreoPreprocessedDataset/choreo_for_usad_normal.csv")#, nrows=1000)
normal = normal.drop(["index" , "is_anomaly" ] , axis = 1)
normal = normal.astype(float)

#Read data
attack = pd.read_csv("Datasets/choreoDataset/choreo_labeled_anomaly_data/CPU_hog/choreo_for_usad_anomaly_CPU_hog.csv")
labels = [label != 0 for label in attack["is_anomaly"].values]
attack = attack.drop(["index", "is_anomaly"], axis=1)
attack = attack.astype(float)

windows_normal=normal.values[ np.arange(window_size)[None, :]+np.arange(normal.shape[0]-window_size)[:, None]]
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
w_size = windows_attack.shape[1]*windows_attack.shape[2] # 12*9 matrix 108
z_size = windows_attack.shape[1]*hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 * windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 * windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(
    data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view([windows_normal_train.shape[0],w_size])
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view([windows_attack.shape[0],w_size])
) , batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

model = UsadModel(w_size, z_size) # see -  stochastic gradient descent
model = to_device(model,device)

history = training(N_EPOCHS,model,train_loader,val_loader)
plot_history(history)

torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model.pth")

checkpoint = torch.load("model.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results,column_wise_results= testing_individual_metrics(model, test_loader, window_size)
print("column shape , " , column_wise_results.shape)
print("results shape , " , results[0].shape)

if(len(results)>1):
    y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                  results[-1].flatten().detach().cpu().numpy()])
else:
    y_pred = results[0].flatten().detach().cpu().numpy()

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))

y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

threshold = np.percentile(y_pred, [83])[0]  # 95th percentile is considered in swat case
# threshold = 0.04291054975241422
print("our threshold: ", threshold)

y_pred_for_eval = []
for val in y_pred:
    if val > threshold:
        y_pred_for_eval.append(1)
    else:
        y_pred_for_eval.append(0)

print("Original evaluation results")
y_pred_for_arr = np.array(y_pred_for_eval)
evaluator = Evaluation(y_test, y_pred_for_arr)
evaluator.print()

print("Neighbourhood adjusted evaluation results")
adjusted_predictions = adjust_predictions_for_neighbourhood(y_test,y_pred_for_arr)
adj_evaluator = Evaluation(y_test, adjusted_predictions)
adj_evaluator.print()

test_set = windows_attack
test_set = test_set.reshape((test_set.shape[0], -1))
test_set = pd.DataFrame(test_set)
test_set["is_anomaly"] = y_test

index_of_maximum_error = torch.argmax(column_wise_results, dim=2)
overall_reconstruction_error = results[0].flatten().detach().cpu().numpy()
k = (window_size - 1) * attack.shape[1]
print(results[0])
print(column_wise_results[0])
metric_value=[]
# index refers to number of metrics
for index in range(9):
    metric_value.append(column_wise_results[0][:, index].flatten().detach().cpu().numpy())

metric_count_histogram(test_set, adjusted_predictions, index_of_maximum_error[0])
plot_multivariate_anomaly(test_set, adjusted_predictions, index_of_maximum_error[0], k)
reconstruction_error_plot(test_set, adjusted_predictions, index_of_maximum_error[0], overall_reconstruction_error,
                          threshold)
reconstruction_error_plot_for_nine_metrics(test_set, adjusted_predictions, index_of_maximum_error[0],
                                           metric_value, threshold)

