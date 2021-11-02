import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import metrics
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib.lines import Line2D

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)
    
def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()
    
def histogram(y_test,y_pred):
    plt.figure(figsize=(12,6))
    plt.hist([y_pred[y_test==0],
              y_pred[y_test==1]],
            bins=20,
            color = ['#82E0AA','#EC7063'],stacked=True)
    plt.title("Results",size=20)
    plt.grid()
    plt.show()
    
def ROC(y_test,y_pred):
    fpr,tpr,tr=roc_curve(y_test,y_pred)
    auc=roc_auc_score(y_test,y_pred)
    idx=np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr,tpr,label="AUC="+str(auc))
    plt.plot(fpr,1-fpr,'r:')
    plt.plot(fpr[idx],tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    return tr[idx]
    
def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted','y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])
    
    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()

def merge(l1, l2):
    return [str(a) + "=" + str(b) for (a, b) in zip(l1, l2)]

def plot_multivariate_anomaly(test_set, predictions,col,col_max_error, k):

    test_set["Index"] = range(test_set.shape[0])
    test_set['predictions'] = predictions
    test_set["max_value"] = col
    test_set['col_max_error'] = col_max_error

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9)
    l = ['ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax9']
    features = ['in_avg_response_time', 'in_throughput', 'in_progress_requests',
                'http_error_count', 'ballerina_error_count', 'cpu', 'memory',
                'cpuPercentage', 'memoryPercentage']

    print(k)
    for n in range(len(l)):
        i = l[n]
        eval(i).plot(test_set.loc[:, k+n], linewidth=1)
        eval(i).set_ylabel(features[n])

    tp_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)]
    fp_index_list = test_set.index[(test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)]
    fn_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)]

    tp_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)
    fp_condition = (test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)
    fn_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)
    condition_analysis = (tp_condition==1) | (fp_condition==1) | (fn_condition==1)

    for n in range(len(l)):
        i = l[n]
        df_vals_tp = test_set.loc[tp_index_list, k+n]
        df_vals_tp_list = df_vals_tp.to_list()
        df_vals_fn = test_set.loc[fn_index_list, k+n]
        df_vals_fn_list = df_vals_fn.to_list()
        df_vals_fp = test_set.loc[fp_index_list, k+n]
        df_vals_fp_list = df_vals_fp.to_list()

        combined_list_tp = merge(tp_index_list, df_vals_tp_list)
        combined_list_fn = merge(fn_index_list, df_vals_fn_list)
        combined_list_fp = merge(fp_index_list, df_vals_fp_list)

        for element in combined_list_tp:
            eval(i).scatter([float(element.split('=')[0])], [float(element.split('=')[1])], c='red',
                            marker='|')  # , s=1000)
        for element in combined_list_fn:
            eval(i).scatter([float(element.split('=')[0])], [float(element.split('=')[1])], c='orange',
                            marker='|')  # , s=1000)
        for element in combined_list_fp:
            eval(i).scatter([float(element.split('=')[0])], [float(element.split('=')[1])], c='green',
                            marker='|')  # , s=1000)

        column_list_name = ['column_0_list', 'column_1_list', 'column_2_list', 'column_3_list', 'column_4_list',
                            'column_5_list', 'column_6_list', 'column_7_list', 'column_8_list']
        df_vals_column_name = ['df_vals_column_0','df_vals_column_1','df_vals_column_2','df_vals_column_3','df_vals_column_4','df_vals_column_5','df_vals_column_6','df_vals_column_7','df_vals_column_8']
        df_vals_column_name_list = ['df_vals_column_0_list','df_vals_column_1_list','df_vals_column_2_list','df_vals_column_3_list','df_vals_column_4_list','df_vals_column_5_list','df_vals_column_6_list','df_vals_column_7_list','df_vals_column_8_list']
        combined_list_col = ['combined_list_column_0','combined_list_column_1','combined_list_column_2','combined_list_column_3','combined_list_column_4','combined_list_column_5','combined_list_column_6','combined_list_column_7','combined_list_column_8']

        for p in range(9):
            column_list_name[p] = test_set.index[(test_set.max_value == p) & (condition_analysis == 1)]
            df_vals_column_name[p] = test_set.iloc[column_list_name[p], k+n]
            df_vals_column_name_list[p] = df_vals_column_name[p].values
            combined_list_col[p] = merge(column_list_name[p], df_vals_column_name_list[p])

            if p == 0:
                for element in combined_list_col[0]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='blue',
                                marker='|', s=50)
            elif p == 1:
                for element in combined_list_col[1]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='yellow',
                                marker='|', s=40)
            elif p == 2:
                for element in combined_list_col[2]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='purple',
                                marker='|', s=40)
            elif p == 3:
                for element in combined_list_col[3]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='brown',
                                marker='|', s=50)
            elif p == 4:
                for element in combined_list_col[4]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='limegreen',
                                marker='|', s=40)
            elif p == 5:
                for element in combined_list_col[5]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='gray',
                                marker='|', s=50)
            elif p == 6:
                for element in combined_list_col[6]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='olive',
                                marker='|', s=40)
            elif p == 7:
                for element in combined_list_col[7]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='crimson',
                                marker='|', s=50)
            elif p == 8:
                for element in combined_list_col[8]:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c='teal',
                                marker='|', s=50)
            eval(i).set_ylim([-0.7, 2.5])

    legend_elements = [
        Line2D([0], [0], marker='|', color='r', label='True Positives', markersize=10),
        Line2D([0], [0], marker='|', color='g', label='False Positives', markersize=10),
        Line2D([0], [0], marker='|', color='orange', label='False Negatives', markersize=10),

        Line2D([0], [0], marker='|', color='blue', label='in_avg_response_time', markersize=10),
        Line2D([0], [0], marker='|', color='yellow', label='in_throughput',
                markersize=10),
        Line2D([0], [0], marker='|', color='purple', label='in_progress_requests', markersize=10),
        Line2D([0], [0], marker='|', color='brown', label='http_error_count',
               markersize=10),
        Line2D([0], [0], marker='|', color='limegreen', label='ballerina_error_count', markersize=10),
        Line2D([0], [0], marker='|', color='gray', label='cpu',
               markersize=10),
        Line2D([0], [0], marker='|', color='olive', label='memory', markersize=10),
        Line2D([0], [0], marker='|', color='crimson', label='cpuPercentage',
               markersize=10),
        Line2D([0], [0], marker='|', color='teal', label='memoryPercentage', markersize=10),
                       ]

    fig.legend(handles=legend_elements, fontsize=10)
    fig.set_size_inches(12, 25)
    fig.tight_layout(pad=10.0)
    plt.savefig("plots_CPU_hog/user_surge_all_metric_plot.png", dpi=200)

class Evaluation:
    def __init__(self, y_test, predict_test):
        self.accuracy = metrics.accuracy_score(y_test, predict_test)
        self.precision = metrics.precision_score(y_test, predict_test)
        self.recall = metrics.recall_score(y_test, predict_test)
        self.auc = metrics.roc_auc_score(y_test, predict_test)
        self.f1_score = metrics.f1_score(y_test, predict_test)
        self.cm = metrics.confusion_matrix(y_test, predict_test)

    def print(self):
        print("Accuracy\tPrecision\tRecall\tAUC\tF1")
        print("%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (self.accuracy, self.precision, self.recall, self.auc, self.f1_score))
        print("Confusion Matrix")
        print(self.cm)

    def obtain_vals(self):
        return (self.accuracy, self.precision, self.recall, self.auc, self.f1_score)


def adjust_predictions_for_neighbourhood(y_test, predict_test, slack=5):
    length = len(y_test)
    adjusted_forecasts = np.copy(predict_test)
    for i in range(length):
        if y_test[i] == predict_test[i]:
            adjusted_forecasts[i] = predict_test[i]
        elif predict_test[i] == 1:  # FP
            if np.sum(y_test[i - slack:i + slack]) > 0:
                # print(y_test[i - slack:i + slack], "=", np.sum(y_test[i - slack:i + slack]))
                adjusted_forecasts[i] = 0  # there is anomaly within 20 in actual, so 1 OK
        elif predict_test[i] == 0:  # FN
            if np.sum(predict_test[i - slack:i + slack]) > 0:
                # print(predict_test[i - slack:i + slack], "=", np.sum(predict_test[i - slack:i + slack]))
                adjusted_forecasts[i] = 1  # there is anomaly within 20 in predicted, so OK
    return adjusted_forecasts


def metric_count_histogram(test_set, predictions,column_max_value):

    test_set['predictions'] = predictions
    test_set["max_value"] = column_max_value

    tp_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)
    fp_condition = (test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)
    fn_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)
    condition_analysis = (tp_condition == 1) | (fp_condition == 1) | (fn_condition == 1)

    col_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    col_1_list = test_set.index[(test_set.max_value == col_list[0]) & (condition_analysis == 1)]
    col_1_list_list =col_1_list.unique()
    col_2_list = test_set.index[(test_set.max_value == col_list[1]) & (condition_analysis == 1)]
    col_2_list_list = col_2_list.unique()
    col_3_list = test_set.index[(test_set.max_value == col_list[2]) & (condition_analysis == 1)]
    col_3_list_list = col_3_list.unique()
    col_4_list = test_set.index[(test_set.max_value == col_list[3]) & (condition_analysis == 1)]
    col_4_list_list = col_4_list.unique()
    col_5_list = test_set.index[(test_set.max_value == col_list[4]) & (condition_analysis == 1)]
    col_5_list_list = col_5_list.unique()
    col_6_list = test_set.index[(test_set.max_value == col_list[5]) & (condition_analysis == 1)]
    col_6_list_list = col_6_list.unique()
    col_7_list = test_set.index[(test_set.max_value == col_list[6]) & (condition_analysis == 1)]
    col_7_list_list = col_7_list.unique()
    col_8_list = test_set.index[(test_set.max_value == col_list[7]) & (condition_analysis == 1)]
    col_8_list_list = col_8_list.unique()
    col_9_list = test_set.index[(test_set.max_value == col_list[8]) & (condition_analysis == 1)]
    col_9_list_list = col_9_list.unique()

    return col_1_list_list,col_2_list_list,col_3_list_list,col_4_list_list,col_5_list_list,col_6_list_list,col_7_list_list,col_8_list_list,col_9_list_list

def reconstruction_error_plot(test_set, predictions,max_value,results,threshold_line):

    test_set['predictions'] = predictions
    test_set['results'] = results
    test_set["max_value"] = max_value
    fig = plt.figure()
    plt.plot(test_set.iloc[:, -1], linewidth = 1)

    tp_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)]
    fp_index_list = test_set.index[(test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)]
    fn_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)]

    tp_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)
    fp_condition = (test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)
    fn_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)
    condition_analysis = (tp_condition == 1) | (fp_condition == 1) | (fn_condition == 1)

    df_vals_tp = test_set.iloc[tp_index_list, -1]
    df_vals_tp_list = df_vals_tp.to_list()
    df_vals_fn = test_set.iloc[fn_index_list, -1]
    df_vals_fn_list = df_vals_fn.to_list()
    df_vals_fp = test_set.iloc[fp_index_list, -1]
    df_vals_fp_list = df_vals_fp.to_list()

    combined_list_tp = merge(tp_index_list, df_vals_tp_list)
    combined_list_fn = merge(fn_index_list, df_vals_fn_list)
    combined_list_fp = merge(fp_index_list, df_vals_fp_list)

    for element in combined_list_tp:
        plt.scatter([float(element.split('=')[0])], [float(element.split('=')[1])], c='red',
                        marker='|')  # , s=1000)
    for element in combined_list_fn:
        plt.scatter([float(element.split('=')[0])], [float(element.split('=')[1])], c='orange',
                        marker='|')  # , s=1000)
    for element in combined_list_fp:
        plt.scatter([float(element.split('=')[0])], [float(element.split('=')[1])], c='green',
                        marker='|')  # , s=1000)

    column_list_name = ['column_0_list', 'column_1_list', 'column_2_list', 'column_3_list', 'column_4_list',
                        'column_5_list', 'column_6_list', 'column_7_list', 'column_8_list']
    df_vals_column_name = ['df_vals_column_0', 'df_vals_column_1', 'df_vals_column_2', 'df_vals_column_3',
                           'df_vals_column_4', 'df_vals_column_5', 'df_vals_column_6', 'df_vals_column_7',
                           'df_vals_column_8']
    df_vals_column_name_list = ['df_vals_column_0_list', 'df_vals_column_1_list', 'df_vals_column_2_list',
                                'df_vals_column_3_list', 'df_vals_column_4_list', 'df_vals_column_5_list',
                                'df_vals_column_6_list', 'df_vals_column_7_list', 'df_vals_column_8_list']
    combined_list_col = ['combined_list_column_0', 'combined_list_column_1', 'combined_list_column_2',
                         'combined_list_column_3', 'combined_list_column_4', 'combined_list_column_5',
                         'combined_list_column_6', 'combined_list_column_7', 'combined_list_column_8']

    for i in range(9):
        column_list_name[i] = test_set.index[(test_set.max_value == i) & (condition_analysis == 1)]
        df_vals_column_name[i] = test_set.iloc[column_list_name[i], -1]
        df_vals_column_name_list[i] = df_vals_column_name[i].values
        combined_list_col[i] = merge(column_list_name[i], df_vals_column_name_list[i])

        if i == 0:
            for element in combined_list_col[0]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='blue',
                                marker='|', s=50)
        elif i == 1:
            for element in combined_list_col[1]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='yellow',
                                marker='|', s=40)
        elif i == 2:
            for element in combined_list_col[2]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='purple',
                                marker='|', s=40)
        elif i == 3:
            for element in combined_list_col[3]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='brown',
                                marker='|', s=50)
        elif i == 4:
            for element in combined_list_col[4]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='limegreen',
                                marker='|', s=40)
        elif i == 5:
            for element in combined_list_col[5]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='gray',
                                marker='|', s=50)
        elif i == 6:
            for element in combined_list_col[6]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='olive',
                                marker='|', s=40)
        elif i == 7:
            for element in combined_list_col[7]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='crimson',
                                marker='|', s=50)
        elif i == 8:
            for element in combined_list_col[8]:
                plt.scatter([float(element.split('=')[0])], [-0.1], c='teal',
                                marker='|', s=50)

    legend_elements = [
        Line2D([0], [0], marker='|', color='r', label='True Positives', markersize=10),
        Line2D([0], [0], marker='|', color='g', label='False Positives', markersize=10),
        Line2D([0], [0], marker='|', color='orange', label='False Negatives', markersize=10),

        Line2D([0], [0], marker='|', color='blue', label='in_avg_response_time', markersize=10),
        Line2D([0], [0], marker='|', color='yellow', label='in_throughput',
                markersize=10),
        Line2D([0], [0], marker='|', color='purple', label='in_progress_requests', markersize=10),
        Line2D([0], [0], marker='|', color='brown', label='http_error_count',
               markersize=10),
        Line2D([0], [0], marker='|', color='limegreen', label='ballerina_error_count', markersize=10),
        Line2D([0], [0], marker='|', color='gray', label='cpu',
               markersize=10),
        Line2D([0], [0], marker='|', color='olive', label='memory', markersize=10),
        Line2D([0], [0], marker='|', color='crimson', label='cpuPercentage',
               markersize=10),
        Line2D([0], [0], marker='|', color='teal', label='memoryPercentage', markersize=10),
                       ]

    fig.legend(handles=legend_elements, fontsize=10)
    plt.axhline(y=threshold_line, color='r', linestyle='-')
    fig.set_size_inches(20, 4)
    fig.tight_layout(pad=10.0)
    plt.savefig("plots_CPU_hog/user_surge_full_reconstruction_error_adjusted.png", dpi=200)

def reconstruction_error_plot_for_nine_metrics(test_set, predictions,col, metric1,metric2,metric3,metric4,metric5,metric6,metric7,metric8,metric9, threshold):

    test_set["Index"] = range(test_set.shape[0])
    test_set['predictions'] = predictions
    test_set["max_value"] = col

    fig, axs = plt.subplots()

    # metric1 = in_avg_response_time
    # metric2 = in_throughput
    # metric3 = in_progress_requests
    # metric4 = http_error_count
    # metric5 = ballerina_error_count
    # metric6 = cpu
    # metric7 = memory
    # metric8 = cpuPercentage
    # metric9 = memoryPercentage

    l = ['ax1', 'ax2', 'ax3', 'ax4', 'ax5', 'ax6', 'ax7', 'ax8', 'ax9']
    axis_colors = ['blue','yellow','purple','brown','limegreen','gray','olive','crimson','teal']
    metrics = ['metric1','metric2','metric3','metric4','metric5','metric6','metric7','metric8','metric9']

    axs.plot(metric1, color='blue', linewidth=1)
    axs.plot(metric2, color='yellow', linewidth=1)
    axs.plot(metric3, color='purple', linewidth=1)
    axs.plot(metric4, color='brown', linewidth=1)
    axs.plot(metric5, color='limegreen', linewidth=1)
    axs.plot(metric6, color='gray', linewidth=1)
    axs.plot(metric7, color='olive', linewidth=1)
    axs.plot(metric8, color='crimson', linewidth=1)
    axs.plot(metric9, color='teal', linewidth=1)

    tp_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)]
    fp_index_list = test_set.index[(test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)]
    fn_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)]

    tp_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)
    fp_condition = (test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)
    fn_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)
    condition_analysis = (tp_condition == 1) | (fp_condition == 1) | (fn_condition == 1)

    df_vals_tp = test_set.iloc[tp_index_list, -1]
    df_vals_tp_list = df_vals_tp.to_list()
    df_vals_fn = test_set.iloc[fn_index_list, -1]
    df_vals_fn_list = df_vals_fn.to_list()
    df_vals_fp = test_set.iloc[fp_index_list, -1]
    df_vals_fp_list = df_vals_fp.to_list()

    combined_list_tp = merge(tp_index_list, df_vals_tp_list)
    combined_list_fn = merge(fn_index_list, df_vals_fn_list)
    combined_list_fp = merge(fp_index_list, df_vals_fp_list)

    for element in combined_list_tp:
        axs.scatter([float(element.split('=')[0])],[-0.05], c='red',
                            marker='|')  # , s=1000)
    for element in combined_list_fn:
        axs.scatter([float(element.split('=')[0])],[-0.05], c='orange',
                            marker='|')  # , s=1000)
    for element in combined_list_fp:
        axs.scatter([float(element.split('=')[0])],[-0.05], c='green',
                            marker='|')  # , s=1000)

    column_list_name = ['column_0_list', 'column_1_list', 'column_2_list', 'column_3_list', 'column_4_list',
                            'column_5_list', 'column_6_list', 'column_7_list', 'column_8_list']
    df_vals_column_name = ['df_vals_column_0', 'df_vals_column_1', 'df_vals_column_2', 'df_vals_column_3',
                               'df_vals_column_4', 'df_vals_column_5', 'df_vals_column_6', 'df_vals_column_7',
                               'df_vals_column_8']
    df_vals_column_name_list = ['df_vals_column_0_list', 'df_vals_column_1_list', 'df_vals_column_2_list',
                                    'df_vals_column_3_list', 'df_vals_column_4_list', 'df_vals_column_5_list',
                                    'df_vals_column_6_list', 'df_vals_column_7_list', 'df_vals_column_8_list']
    combined_list_col = ['combined_list_column_0', 'combined_list_column_1', 'combined_list_column_2',
                             'combined_list_column_3', 'combined_list_column_4', 'combined_list_column_5',
                             'combined_list_column_6', 'combined_list_column_7', 'combined_list_column_8']

    for p in range(9):
        column_list_name[p] = test_set.index[(test_set.max_value == p) & (condition_analysis == 1)]
        df_vals_column_name[p] = test_set.iloc[column_list_name[p], -1]
        df_vals_column_name_list[p] = df_vals_column_name[p].values
        combined_list_col[p] = merge(column_list_name[p], df_vals_column_name_list[p])

        if p == 0:
            for element in combined_list_col[0]:
                    axs.scatter([float(element.split('=')[0])], [-0.1], c='blue',
                                    marker='|', s=50)
        elif p == 1:
            for element in combined_list_col[1]:
                    axs.scatter([float(element.split('=')[0])], [-0.1], c='yellow',
                                    marker='|', s=40)
        elif p == 2:
            for element in combined_list_col[2]:
                    axs.scatter([float(element.split('=')[0])], [-0.1], c='purple',
                                    marker='|', s=40)
        elif p == 3:
            for element in combined_list_col[3]:
                axs.scatter([float(element.split('=')[0])], [-0.1], c='brown',
                                    marker='|', s=50)
        elif p == 4:
            for element in combined_list_col[4]:
                axs.scatter([float(element.split('=')[0])], [-0.1], c='limegreen',
                                    marker='|', s=40)
        elif p == 5:
            for element in combined_list_col[5]:
                axs.scatter([float(element.split('=')[0])], [-0.1], c='gray',
                                    marker='|', s=50)
        elif p == 6:
            for element in combined_list_col[6]:
                axs.scatter([float(element.split('=')[0])], [-0.1], c='olive',
                                    marker='|', s=40)
        elif p == 7:
            for element in combined_list_col[7]:
                axs.scatter([float(element.split('=')[0])], [-0.1], c='crimson',
                                    marker='|', s=50)
        elif p == 8:
            for element in combined_list_col[8]:
                axs.scatter([float(element.split('=')[0])], [-0.1], c='teal',
                                    marker='|', s=50)


    legend_elements = [
        Line2D([0], [0], marker='|', color='r', label='True Positives', markersize=10),
        Line2D([0], [0], marker='|', color='g', label='False Positives', markersize=10),
        Line2D([0], [0], marker='|', color='orange', label='False Negatives', markersize=10),

        Line2D([0], [0], marker='|', color='blue', label='in_avg_response_time', markersize=10),
        Line2D([0], [0], marker='|', color='yellow', label='in_throughput',
                markersize=10),
        Line2D([0], [0], marker='|', color='purple', label='in_progress_requests', markersize=10),
        Line2D([0], [0], marker='|', color='brown', label='http_error_count',
               markersize=10),
        Line2D([0], [0], marker='|', color='limegreen', label='ballerina_error_count', markersize=10),
        Line2D([0], [0], marker='|', color='gray', label='cpu',
               markersize=10),
        Line2D([0], [0], marker='|', color='olive', label='memory', markersize=10),
        Line2D([0], [0], marker='|', color='crimson', label='cpuPercentage',
               markersize=10),
        Line2D([0], [0], marker='|', color='teal', label='memoryPercentage', markersize=10),
                       ]

    fig.legend(handles=legend_elements, fontsize=10)
    plt.axhline(threshold, color='red', linewidth=1)
    fig.set_size_inches(20, 13)
    fig.tight_layout(pad=10.0)
    plt.savefig("plots_CPU_hog/user_surge_reconstruction_error_all_metric.png", dpi=200)










