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

def function_for_condition(test_set,predictions) :
    test_set['predictions'] = predictions
    # tp refers to true positives
    # fp refers to false positives
    # fn refers to false negatives
    tp_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)
    fp_condition = (test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)
    fn_condition = (test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)
    condition_analysis = (tp_condition == 1) | (fp_condition == 1) | (fn_condition == 1)

    return condition_analysis

def legend_elements():
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
    return legend_elements


def plot_multivariate_anomaly(test_set, predictions, index_of_maximum_error, k):
    """ Time series plot for all the metrics """

    test_set['predictions'] = predictions
    test_set["index_of_maximum_error"] = index_of_maximum_error

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

        for index in range(9):
            color_for_metric = ['blue', 'yellow', 'purple', 'brown', 'limegreen', 'gray', 'olive', 'crimson', 'teal']
            condition_for_metric = test_set.index[(test_set.index_of_maximum_error == index) & (function_for_condition(test_set,predictions) == 1)]
            df_column = test_set.iloc[condition_for_metric, -1]
            df_column_values = df_column.to_list()
            combined_list_column = merge(condition_for_metric, df_column_values)

            if index == index:
                for element in combined_list_column:
                    eval(i).scatter([float(element.split('=')[0])], [-0.5], c=color_for_metric[index],
                                    marker='|', s=50)
            eval(i).set_ylim([-0.7, 2.5])

    fig.legend(handles=legend_elements(), fontsize=10)
    fig.set_size_inches(12, 25)
    fig.tight_layout(pad=10.0)
    plt.savefig("backend_failure_multivariate_anomaly_plot.png", dpi=200)


def metric_count_histogram(test_set, predictions, index_of_maximum_error):
    """ The total count of metrics on how many times it has been the major deviating feature in the time frame
    has been taken and plotted in a bar chart """

    test_set['predictions'] = predictions
    test_set["index_of_maximum_error"] = index_of_maximum_error

    features = ('in_avg_response_time', 'in_throughput', 'in_progress_requests',
                'http_error_count', 'ballerina_error_count', 'cpu', 'memory',
                'cpuPercentage', 'memoryPercentage')
    y_value = np.arange(len(features))

    performance=[]
    for index in range(9):
        col_list = test_set.index[(test_set.index_of_maximum_error == index) & (function_for_condition(test_set,predictions) == 1)]
        col_list_list = len(col_list.unique())
        performance.append(col_list_list)

    plt.bar(y_value, performance, align='center', alpha=0.5)
    plt.xticks(y_value , features)
    plt.xticks(fontsize=3)
    plt.ylabel('Value')
    plt.title('backend_failure')
    plt.savefig("backend_failure_metric_count_histogram.png", dpi=200)

def reconstruction_error_plot(test_set, predictions, index_of_maximum_error, results, threshold_line):
    """ This plot depicts the overall reconstruction error """

    test_set['predictions'] = predictions
    test_set['index_of_maximum_error'] = index_of_maximum_error
    test_set['results'] = results
    fig = plt.figure()

    plt.plot(test_set.loc[:, ['results']], linewidth = 1)

    tp_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)]
    fp_index_list = test_set.index[(test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)]
    fn_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)]

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

    for index in range(9):
        color_for_metric = ['blue', 'yellow', 'purple', 'brown', 'limegreen', 'gray', 'olive', 'crimson', 'teal']
        condition_for_metric = test_set.index[(test_set.index_of_maximum_error == index) & (function_for_condition(test_set, predictions) == 1)]
        df_column = test_set.iloc[condition_for_metric, -1]
        df_column_values = df_column.to_list()
        combined_list_column = merge(condition_for_metric, df_column_values)

        if index == index:
            for element in combined_list_column:
                plt.scatter([float(element.split('=')[0])], [-0.5], c=color_for_metric[index],
                            marker='|', s=50)

    fig.legend(handles=legend_elements(), fontsize=10)
    plt.axhline(y=threshold_line, color='r', linestyle='-')
    fig.set_size_inches(20, 4)
    fig.tight_layout(pad=10.0)
    plt.savefig("backend_failure_overall_reconstruction_error_plot.png", dpi=200)

def reconstruction_error_plot_for_nine_metrics(test_set, predictions, index_of_maximum_error, metrics, threshold):
    """ This plot depicts the metric wise reconstruction error """

    test_set['predictions'] = predictions
    test_set['index_of_maximum_error'] = index_of_maximum_error
    fig, axs = plt.subplots()

    axis_colors = ['blue', 'yellow', 'purple', 'brown', 'limegreen', 'gray', 'olive', 'crimson', 'teal']
    for i in range(9):
        axs.plot(metrics[i], color=axis_colors[i], linewidth=1)

    tp_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 1)]
    fp_index_list = test_set.index[(test_set['is_anomaly'] == 0) & (test_set['predictions'] == 1)]
    fn_index_list = test_set.index[(test_set['is_anomaly'] == 1) & (test_set['predictions'] == 0)]

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

    # index refers to the number of metrics considered
    for index in range(9):
        color_for_metric = ['blue', 'yellow', 'purple', 'brown', 'limegreen', 'gray', 'olive', 'crimson', 'teal']
        condition_for_metric = test_set.index[(test_set.index_of_maximum_error == index) & (function_for_condition(test_set,predictions) == 1)]
        df_column = test_set.iloc[condition_for_metric, -1]
        df_column_values = df_column.to_list()
        combined_list_column = merge(condition_for_metric, df_column_values)

        if index == index:
            for element in combined_list_column:
                plt.scatter([float(element.split('=')[0])], [-0.1], c=color_for_metric[index],
                            marker='|', s=50)

    fig.legend(handles=legend_elements(), fontsize=10)
    plt.axhline(threshold, color='red', linewidth=1)
    fig.set_size_inches(20, 13)
    fig.tight_layout(pad=10.0)
    plt.savefig("backend_failure_metric_level_reconstruction_error_plot.png", dpi=200)
    
