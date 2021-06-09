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
                     opt_drive_path='data/optical_drive_table.csv'):

    df = read_table(path, index_col=0)
    df.drop(columns='Comments (don\'t use commas)', inplace=True)

    cpu_df = read_table(path=cpu_path)
    cpu_map_dict = {}
    for cpu in df['CPU'].unique():
        cpu_map_dict[cpu] = cpu_df[cpu_df['CPU Name'] == cpu].iloc[0]['CPU Mark']

    cpu_marks = cpu_df['CPU Mark']
    df['CPU'] = df['CPU'].map(cpu_map_dict)

    gpu_df = read_table(path=gpu_path)
    gpu_map_dict = {}
    for gpu in df['GPU'].unique():
        gpu_map_dict[gpu] = gpu_df[gpu_df['GPU Name'] == gpu].iloc[0]['Benchmark']

    gpu_marks = gpu_df['Benchmark']
    df['GPU'] = df['GPU'].map(gpu_map_dict)

    transformations = {column: {'log2': False, 'scaler': None} for column in df.columns}
    for log_column in ['RAM (GB)', 'SSD (GB)', 'HDD (GB)']:
        df[log_column] = np.log2(df[log_column], where=df[log_column] != 0)
        transformations[log_column]['log2'] = True

    df['Primary use'] = df['Primary use'].map({'Home': 0,
                                               'Work': 1,
                                               'Production': 2,
                                               'Programming': 3,
                                               'ML': 4,
                                               'Gaming': 5})

    ram_df = read_table(ram_path)
    ram_values = np.log2(ram_df['Capacity'], where=ram_df['Capacity'] != 0)
    ssd_df = read_table(ssd_path)
    ssd_values = np.log2(ssd_df['Capacity'], where=ssd_df['Capacity'] != 0)
    hdd_df = read_table(hdd_path)
    hdd_values = np.log2(hdd_df['Capacity'], where=hdd_df['Capacity'] != 0)
    opt_drive_df = read_table(opt_drive_path)
    opt_drive_values = opt_drive_df['Boolean State']

    for column in df.columns:
        mmscaler = MinMaxScaler()
        if column == 'CPU':
            # we need to fit with all the possible benchmarks, not only the ones in the case base
            mmscaler.fit(cpu_marks.to_numpy().reshape(-1, 1))
            df[column] = mmscaler.transform(df[column].to_numpy().reshape(-1, 1))
        elif column == 'GPU':
            # we need to fit with all the possible benchmarks, not only the ones in the case base
            mmscaler.fit(gpu_marks.to_numpy().reshape(-1, 1))
            df[column] = mmscaler.transform(df[column].to_numpy().reshape(-1, 1))
        elif column == 'RAM (GB)':
            # we need to fit with all the possible benchmarks, not only the ones in the case base
            mmscaler.fit(ram_values.to_numpy().reshape(-1, 1))
            df[column] = mmscaler.transform(df[column].to_numpy().reshape(-1, 1))
        elif column == 'SSD (GB)':
            # we need to fit with all the possible benchmarks, not only the ones in the case base
            mmscaler.fit(ssd_values.to_numpy().reshape(-1, 1))
            df[column] = mmscaler.transform(df[column].to_numpy().reshape(-1, 1))
        elif column == 'HDD (GB)':
            # we need to fit with all the possible benchmarks, not only the ones in the case base
            mmscaler.fit(hdd_values.to_numpy().reshape(-1, 1))
            df[column] = mmscaler.transform(df[column].to_numpy().reshape(-1, 1))
        elif column == 'Optical Drive (1 = DVD, 0 = None)':
            # we need to fit with all the possible benchmarks, not only the ones in the case base
            mmscaler.fit(opt_drive_values.to_numpy().reshape(-1, 1))
            df[column] = mmscaler.transform(df[column].to_numpy().reshape(-1, 1))
        else:
            df[column] = mmscaler.fit_transform(df[column].to_numpy().reshape(-1, 1))

        transformations[column]['scaler'] = mmscaler

    return df, transformations
