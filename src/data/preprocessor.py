import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def read_table(path='data/cpu_table.csv', **kwargs):
    return pd.read_csv(path, **kwargs)


def read_initial_cbl(path='data/pc_specs.csv', 
                     cpu_path='data/cpu_table.csv',
                     gpu_path='data/gpu_table.csv',
                     ram_path='data/ram_table.csv',
                     ssd_path='data/ssd_table.csv',
                     hdd_path='data/hdd_table.csv',
                     opt_drive_path='data/optical_drive_table.csv',
                     feature_scalers_meta='data/feature_scalers.json'):

    df = read_table(path, index_col=0)
    df.drop(columns='Comments (don\'t use commas)', inplace=True)

    cpu_df = read_table(path=cpu_path)
    cpu_map_dict = {}
    for cpu in df['CPU'].unique():
        cpu_map_dict[cpu] = cpu_df[cpu_df['CPU Name'] == cpu].iloc[0]['CPU Mark']

    df['CPU'] = df['CPU'].map(cpu_map_dict)

    gpu_df = read_table(path=gpu_path)
    gpu_map_dict = {}
    for gpu in df['GPU'].unique():
        gpu_map_dict[gpu] = gpu_df[gpu_df['GPU Name'] == gpu].iloc[0]['Benchmark']

    df['GPU'] = df['GPU'].map(gpu_map_dict)

    transformations = {column: {'log2': False, 'scaler': None} for column in df.columns}
    feature_scalers_meta = json.load(open(feature_scalers_meta))

    for feature, attributes in feature_scalers_meta.items():
        transformations[feature]['log2'] = attributes['log2']
        min_value, max_value = attributes['min'], attributes['max']

        if 'map' in attributes:
            df[feature] = df[feature].map(attributes['map'])
            transformations[feature]['map'] = attributes['map']

        if attributes['log2']:
            min_value, max_value = np.log2(min_value+1), np.log2(max_value+1)
            df[feature] = np.log2(df[feature] + 1)

        transformations[feature]['scaler'] = MinMaxScaler().fit(
            np.array([min_value, max_value]).reshape(-1, 1)
        )
        df[feature] = transformations[feature]['scaler'].transform(
            df[feature].to_numpy().reshape(-1, 1)
        )
    return df, transformations
