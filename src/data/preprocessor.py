import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def read_cpu_table(path='data/cpu_table.csv'):
    return pd.read_csv(path)


def read_gpu_table(path='data/gpu_table.csv'):
    return pd.read_csv(path)


def read_initial_cbl(path='data/pc_specs.csv', 
                     gpu_path='data/gpu_table.csv', 
                     cpu_path='data/cpu_table.csv'):
    df = pd.read_csv(path)
    df.drop(columns=['ID', 'Comments (don\'t use commas)'], inplace=True)

    cpu_df = read_cpu_table(path=cpu_path)
    cpu_map_dict = {}
    for cpu in df['CPU'].unique():
        cpu_map_dict[cpu] = cpu_df[cpu_df['CPU Name'] == cpu].iloc[0]['CPU Mark']

    df['CPU'] = df['CPU'].map(cpu_map_dict)

    gpu_df = read_gpu_table(path=gpu_path)
    gpu_map_dict = {}
    for gpu in df['GPU'].unique():
        gpu_map_dict[gpu] = gpu_df[gpu_df['GPU Name'] == gpu].iloc[0]['Benchmark']

    df['GPU'] = df['GPU'].map(gpu_map_dict)

    for log_column in ['RAM (GB)', 'SSD (GB)', 'HDD (GB)']:
        df[log_column] = np.log2(df[log_column], where=df[log_column] != 0)

    df['Primary use'] = df['Primary use'].map({'Home': 0,
                                               'Work': 1,
                                               'Production': 2,
                                               'Programming': 3,
                                               'ML': 4,
                                               'Gaming': 5})

    for column in df.columns:
        mmscaler = MinMaxScaler()
        df[column] = mmscaler.fit_transform(df[column].to_numpy().reshape(-1, 1))

    return df.to_numpy()
