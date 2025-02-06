import pandas as pd
import os
from datetime import datetime
import pathlib
import datetime
import glob
import configparser

TRAINING_DATA_FILE = 'training_data.csv'


def get_config():
    config = configparser.ConfigParser()
    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    config.read(config_file_path)
    return config


def absolute_path(path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def append_files(processed_files: list) -> pd.DataFrame:
    if len(processed_files) == 0:
        raise Exception('Empty file list!')

    df = factory_ts_df(processed_files.pop())
    for file in processed_files:
        df.append(factory_ts_df(file))
    return df


def factory_data_df(ts_min: datetime.datetime, ts_max: datetime.datetime) -> pd.DataFrame:
    data_file_path = training_data_file_path(ts_min, ts_max)
    if not os.path.isfile(data_file_path):
        prepare_training_data(ts_min=ts_min, ts_max=ts_max)

    return factory_ts_df(data_file_path)


def prepare_training_data(ts_min: datetime.datetime, ts_max: datetime.datetime):
    config = get_config()

    sm_processed_files = preprocess_synopmeteo_data(config)
    ap_processed_files = preprocess_airpointers_data(config)
    ap = append_files(ap_processed_files)
    sm = append_files(sm_processed_files)

    result_df = ap.join(sm)
    result_df = result_df.fillna(method='ffill')
    result_df = result_df[result_df.index >= ts_min]
    result_df = result_df[result_df.index <= ts_max]

    data_file = training_data_file_path(ts_min, ts_max)
    result_df.to_csv(data_file)


def training_data_file_path(ts_min: datetime.datetime, ts_max: datetime.datetime) -> str:
    config = get_config()
    processed_dir = absolute_path(config['merged']['data_dir'])

    data_file = str(ts_min) + '-' + str(ts_max) + '_' + TRAINING_DATA_FILE
    data_file_name = windows_compatible_file_name(data_file)
    return os.path.join(processed_dir, data_file_name)


def factory_ts_df(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def preprocess_airpointers_data(config: configparser.ConfigParser) -> list:
    airpointers_data_dir = absolute_path(config['airpointers']['data_dir'])
    processed_dir = absolute_path(config['airpointers']['processed_files_dir'])
    ap_measurement_params = config['airpointers']['measurement_values'].split(',')

    processed_files = []
    files = glob.glob(os.path.join(airpointers_data_dir, '*.csv'))
    for file in files:
        data_df = pd.read_csv(file, index_col='msr_iso_datetime',
                              usecols=['msr_iso_datetime', 'station_name'].append(ap_measurement_params))
        data_df.index = pd.to_datetime(data_df.index)
        data_df = data_df.dropna(subset=['value_pm10'])
        data_df = data_df[pd.notnull(data_df['value_pm10'])]

        find_duplicates_index = pd.MultiIndex.from_arrays([data_df.index, data_df['station_name']])
        duplicated_index = find_duplicates_index[find_duplicates_index.duplicated()]
        duplicated_index = duplicated_index.get_level_values('msr_iso_datetime')

        filtered_df = data_df[~data_df.index.isin(duplicated_index.values)]

        df = pre_process_data(filtered_df, pivot_col='station_name', measurement_params=ap_measurement_params,
                              shift_values=[3, 6, 9, 12, 24])
        processed_files.append(to_processed_file(df, file, processed_dir))
    return processed_files


def preprocess_synopmeteo_data(config: configparser.ConfigParser) -> list:
    synopmeteo_data_dir = absolute_path(config['synopmeteo']['data_dir'])
    processed_dir = absolute_path(config['synopmeteo']['processed_files_dir'])
    sp_measurement_params = config['synopmeteo']['measurement_values'].split(',')

    processed_files = []
    files = glob.glob(os.path.join(synopmeteo_data_dir, '*.csv'))
    for file in files:
        data_df = pd.read_csv(file, index_col='msr_iso_datetime',
                              usecols=['msr_iso_datetime', 'station_name'].append(sp_measurement_params))
        df = pre_process_data(data_df, pivot_col='station_name', measurement_params=sp_measurement_params,
                              shift_values=[])
        processed_files.append(to_processed_file(df, file, processed_dir))
    return processed_files


def get_column_names(column_names_list: list, add_postfix: str) -> dict:
    renamed_columns = {}
    for column_name in column_names_list:
        renamed_columns[column_name] = column_name + '_' + add_postfix
    return renamed_columns


def add_shifted_ts_values(df: pd.DataFrame, values_df: pd.DataFrame, new_column_names: dict, shift_periods: int,
                          shift_freq: str = 'H') -> pd.DataFrame:
    values_df = values_df.tshift(-shift_periods, shift_freq)
    values_df = values_df.rename(columns=new_column_names)
    values_df.fillna(method='ffill')
    return df.join(values_df)


def pre_process_data(raw_data_df: pd.DataFrame, pivot_col: str, measurement_params: list,
                     shift_values: list) -> pd.DataFrame:
    measurement_timestamps_min = raw_data_df.index.min()
    measurement_timestamps_max = raw_data_df.index.max()
    timestamps_index = pd.date_range(start=measurement_timestamps_min, end=measurement_timestamps_max, freq='H')
    df = pd.DataFrame(index=timestamps_index)

    raw_data_df.drop_duplicates()
    pivoted = raw_data_df.pivot(columns=pivot_col, values=measurement_params)
    for measurement in measurement_params:
        pivot_df = pivoted[measurement]
        pivot_df.index = pd.to_datetime(pivot_df.index)
        measurement_df = pivot_df.rename(columns=get_column_names(pivot_df.columns.to_list(), measurement))
        df = df.join(measurement_df)

        if len(shift_values) > 0:
            for val in shift_values:
                name_suffix = '_+' + str(val)
                df = add_shifted_ts_values(df, measurement_df,
                                           get_column_names(measurement_df.columns.to_list(), name_suffix), val)
        df = df.fillna(method='ffill')
    return df


def to_processed_file(processed_df: pd.DataFrame, processed_file: str, processed_files_dir: str) -> str:
    os.makedirs(processed_files_dir, exist_ok=True)
    file_basename = pathlib.Path(processed_file).stem
    data_file = os.path.join(processed_files_dir, file_basename + '_processed.csv')
    if os.path.exists(data_file):
        processed_file_name = windows_compatible_file_name(
            file_basename + '_' + str(datetime.datetime.now()) + '_processed.csv')

        data_file = os.path.join(processed_files_dir, processed_file_name)
    processed_df.to_csv(data_file, index=True)
    return data_file


def windows_compatible_file_name(file_name: str) -> str:
    file_name = file_name.replace(" ", "_")
    # file_name = file_name.replace(".", "")
    file_name = file_name.replace(":", "")
    return file_name


def trim_df(df: pd.DataFrame, selected_columns=None) -> pd.DataFrame:
    if selected_columns is None:
        selected_columns = df.columns

    df = df.sort_index()
    df[selected_columns] = df[selected_columns].fillna(method='ffill')
    ts_first_all_available = max(df[selected_columns].notna().idxmax())
    df_trimmed = df[df.index >= ts_first_all_available]
    return df_trimmed
