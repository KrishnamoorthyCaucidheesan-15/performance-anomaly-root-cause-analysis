# !rm -r sample_data
# !git clone https://github.com/manigalati/usad
# %cd usad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils

from utils import get_default_device, to_device,Evaluation,adjust_predictions_for_neighbourhood,metric_count_histogram,reconstruction_error_plot,reconstruction_error_plot_for_nine_metrics,plot_multivariate_anomaly
from usad import UsadModel, device,training, testing_individual_metrics

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
# x = normal.values

#Read data
attack = pd.read_csv("Datasets/choreoDataset/choreo_labeled_anomaly_data/CPU_hog/choreo_for_usad_anomaly_echo_service_CPU_hog_spikes.csv")
labels = [label != 0 for label in attack["is_anomaly"].values]
attack = attack.drop(["index", "is_anomaly"], axis = 1)
attack = attack.astype(float)
x = attack.values

windows_normal=normal.values[ np.arange(window_size)[None, :]+np.arange(normal.shape[0]-window_size)[:, None]]
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]
w_size = windows_attack.shape[1]*windows_attack.shape[2] # 12*9 matrix 108
z_size = windows_attack.shape[1]*hidden_size

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

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
model = to_device(model,device) #**

history = training(N_EPOCHS,model,train_loader,val_loader)
# plot_history(history)

torch.save({
            'encoder': model.encoder.state_dict(),
            'decoder1': model.decoder1.state_dict(),
            'decoder2': model.decoder2.state_dict()
            }, "model.pth")

# enc_input = torch.randn([1, w_size])
# dec_input = torch.randn([1, z_size])
#
# torch.onnx.export(model.encoder, enc_input, "./output/enc.onnx")
# torch.onnx.export(model.decoder1, dec_input, "./output/dec1.onnx")
# torch.onnx.export(model.decoder2, dec_input, "./output/dec2.onnx")
#
# checkpoint = torch.load("./output/model.pth")

checkpoint = torch.load("model.pth")

model.encoder.load_state_dict(checkpoint['encoder'])
model.decoder1.load_state_dict(checkpoint['decoder1'])
model.decoder2.load_state_dict(checkpoint['decoder2'])

results,column_wise_results= testing_individual_metrics(model, test_loader, window_size)
print("column shape , " , column_wise_results.shape)
print("results shape , " , results[0].shape)

col_maximum_error_index = torch.argmax(column_wise_results, dim=2)

if(len(results)>1):
    y_pred=np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                  results[-1].flatten().detach().cpu().numpy()])
else:
    y_pred=results[0].flatten().detach().cpu().numpy()

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))

y_test = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels ]

# actual_labels = []
# actual_labels_np_array = y[4:-1].to_numpy()
# y_test = np.concatenate(np.zeros(windows_normal_test.shape[0],actual_labels_np_array))

threshold = np.percentile(y_pred, [94])[0]  # 95th percentile is considered in swat case
# threshold = 0.04291054975241422
print("our threshold: ", threshold)

# col_max= find_max_error(column_wise_results)
# col_max_error =(find_max_error_cause(column_wise_results).values > threshold)
maximum_deviating_metric = (torch.max(column_wise_results,dim=2).values > threshold)
print(results[0])
print(column_wise_results[0])

y_pred_for_eval = []
for val in y_pred:
    if val > threshold:
        y_pred_for_eval.append(1)
    else:
        y_pred_for_eval.append(0)

#list created seperately represent each metric as an axis
ax1=[]
ax2=[]
ax3=[]
ax4=[]
ax5=[]
ax6=[]
ax7=[]
ax8=[]
ax9=[]
axis_number = ["ax_1","ax_2","ax_3","ax_4","ax_5","ax_6","ax_7","ax_8","ax_9"]
axis = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
for m in range(len(column_wise_results[0])):
    if (m) < len(column_wise_results[0]):
        ax_1 = column_wise_results[0][m][0]
        ax_1 = ax_1.flatten().detach().cpu().numpy()
        ax_2 = column_wise_results[0][m][1]
        ax_2 = ax_2.flatten().detach().cpu().numpy()
        ax_3 = column_wise_results[0][m][2]
        ax_3 = ax_3.flatten().detach().cpu().numpy()
        ax_4 = column_wise_results[0][m][3]
        ax_4 = ax_4.flatten().detach().cpu().numpy()
        ax_5 = column_wise_results[0][m][4]
        ax_5 = ax_5.flatten().detach().cpu().numpy()
        ax_6 = column_wise_results[0][m][5]
        ax_6 = ax_6.flatten().detach().cpu().numpy()
        ax_7 = column_wise_results[0][m][6]
        ax_7 = ax_7.flatten().detach().cpu().numpy()
        ax_8 = column_wise_results[0][m][7]
        ax_8 = ax_8.flatten().detach().cpu().numpy()
        ax_9 = column_wise_results[0][m][8]
        ax_9 = ax_9.flatten().detach().cpu().numpy()

        ax1.append(ax_1)
        ax2.append(ax_2)
        ax3.append(ax_3)
        ax4.append(ax_4)
        ax5.append(ax_5)
        ax6.append(ax_6)
        ax7.append(ax_7)
        ax8.append(ax_8)
        ax9.append(ax_9)

    else:
        break

print("Original evaluation results")
y_pred_for_arr = np.array(y_pred_for_eval)
evaluator = Evaluation(y_test, y_pred_for_arr)
# evaluator_2 = evaluator.obtain_vals()
# evaluator_3 = evaluator_2[1]
# print(evaluator_2[1])
evaluator.print()

print("Neighbourhood adjusted evaluation results")
adjusted_predictions = adjust_predictions_for_neighbourhood(y_test,y_pred_for_arr)
adj_evaluator = Evaluation(y_test, adjusted_predictions)
adj_evaluator.print()

test_set = windows_attack
test_set = test_set.reshape((test_set.shape[0], -1))
test_set = pd.DataFrame(test_set)
test_set["is_anomaly"] = y_test
# attack_with_time_stamp = pd.read_csv("../dataset/choreo_for_usad_anomaly_with_timestamp.csv")
n = windows_attack.shape[0]
# test_set_with_timestamp = attack_with_time_stamp.iloc[-n - 1:-1, :]
# test_set_with_timestamp['prediction'] = y_pred_for_arr[-n:]
# test_set_with_timestamp['adj_prediction'] = adjusted_predictions[-n:]
# test_set_with_timestamp['test_score'] = y_pred[-n:]
# test_set_with_timestamp['y_test'] = y_test[-n:]
# test_set_with_timestamp.to_csv('usad_predictions.csv')
features = ('in_avg_response_time', 'in_throughput', 'in_progress_requests',
                'http_error_count', 'ballerina_error_count', 'cpu', 'memory',
                'cpuPercentage', 'memoryPercentage')
k = (window_size - 1) * attack.shape[1]
print(col_maximum_error_index[0].shape)
x = metric_count_histogram(test_set,adjusted_predictions,col_maximum_error_index[0])
y_pos = np.arange(len(features))

print(x[0].size)
print(x[1].size)
print(x[2].size)
print(x[3].size)
print(x[4].size)
print(x[5].size)
print(x[6].size)
print(x[7].size)
print(x[8].size)

performance = [x[0].size,x[1].size,x[2].size,x[3].size,x[4].size,x[5].size,x[6].size,x[7].size,x[8].size]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, features)
plt.xticks(fontsize = 3)
plt.ylabel('Value')
plt.title('user_surge')
plt.savefig("plots_CPU_hog/user_surge_metric_count_histogram.png", dpi=200)

reconstructional_error_values = results[0].flatten().detach().cpu().numpy()

# plot_multivariate_anomaly(test_set, adjusted_predictions,col_maximum_error_index[0],maximum_deviating_metric[0], k)
plot_multivariate_anomaly(test_set, adjusted_predictions,col_maximum_error_index[0],maximum_deviating_metric[0], k)

reconstruction_error_plot_for_nine_metrics(test_set, adjusted_predictions,col_maximum_error_index[0],ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,threshold)
# reconstruction_error_plot_for_nine_metrics2(test_set, adjusted_predictions,col_maximum_error_index[0],ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,threshold,k)

# print(plot_multivariate_anomaly_preprocessed_data_analysis(test_set, adjusted_predictions,col_maximum_error_index[0],maximum_deviating_metric[0],threshold, k))
# plot_multivariate_anomaly_preprocessed_data_analysis(test_set,adjusted_predictions,col_maximum_error_index[0],reconstructional_error_values,threshold)
reconstruction_error_plot(test_set,y_pred_for_arr,col_maximum_error_index[0],reconstructional_error_values,threshold)
# print(plot_multivariate_anomaly_reconstruction_error_metrics(test_set,adjusted_predictions,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,threshold))

