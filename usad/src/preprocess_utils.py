import os
import pandas as pd
import random

# Global variables
SEED = 7
TIMESTAMP_COLUMN_NAME = "datetime"

LABEL_COLUMN_NAME = "is_anomaly"
PERCENTILE_VALUE = 99
MAX_VAL = {
    "in_avg_response_time": 10000,
    "in_throughput": 800,
    "in_progress_requests": 50,
    "http_error_count": 20,
    "ballerina_error_count": 20,
    "cpu": 7500,
    "memory": 750,
    "cpuPercentage": 100,
    "memoryPercentage": 100
}


def normalize_dataset(df, max_dict):
    for col in max_dict:
        df[col] = df[col] / max_dict[col]
        df.loc[df[col] > 1, col] = 1
    return df


def get_file_list(path):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.csv')]
    random.seed(SEED)
    random.shuffle(list_of_files)
    return list_of_files


def load_dataset(path):
    file_list = get_file_list(path)

    datasets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_dataset(df, MAX_VAL) for df in datasets]

    data_set = pd.concat(normalized_datasets)
    data_set = data_set.fillna(0)
    X = data_set
    # X = data_set.drop([TIMESTAMP_COLUMN_NAME], axis=1)
    return X


def get_file_list_with_ignore(path, ignore_type):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if
                          (file.endswith('.csv') & (ignore_type not in file))]
    random.shuffle(list_of_files)
    print("Number of files : ", ignore_type, " ", len(list_of_files))
    return list_of_files


def get_file_list_with_keep(path, keep):
    list_of_files = list()
    for (dirpath, dirnames, filenames) in os.walk(path):
        list_of_files += [os.path.join(dirpath, file) for file in filenames if (file.endswith('.csv') & (keep in file))]
    random.shuffle(list_of_files)
    print("Number of files : ", keep, " ", len(list_of_files))
    return list_of_files


def load_dataset_with_ignore(path, ignore):
    file_list = get_file_list_with_ignore(path, ignore)

    datasets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_dataset(df, MAX_VAL) for df in datasets]

    data_set = pd.concat(normalized_datasets)
    data_set = data_set.fillna(0)
    X = data_set
    # X = data_set.drop([TIMESTAMP_COLUMN_NAME], axis=1)
    return X


def load_dataset_with_keep(path, keep):
    file_list = get_file_list_with_keep(path, keep)

    datasets = [pd.read_csv(f) for f in file_list]
    normalized_datasets = [normalize_dataset(df, MAX_VAL) for df in datasets]

    data_set = pd.concat(normalized_datasets)
    data_set = data_set.fillna(0)
    X = data_set
    # X = data_set.drop([TIMESTAMP_COLUMN_NAME], axis=1)
    return X