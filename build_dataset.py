"""
Given config -> TaskDataset for train/test/val

"""
from datasets.features.features import ClassLabel
from datasets import load_dataset
from Config import DataConfig

from pandas.core.frame import DataFrame
from dataset_utils import set_imb_count_dict

import pandas as pd


class TaskDataset:
    def __init__(self, data: DataFrame, label_feature: ClassLabel, config: DataConfig):
        self.info = config
        self.data = data[[text_field for text_field in config.text_fields]].agg('\n'.join, axis=1)
        self.labels = data[config.label_field]
        self.label_feature = label_feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data.iloc[i]
        label = self.labels.iloc[i]

        return data, label


def split_df(df: DataFrame,
             label_features: ClassLabel,
             config: DataConfig):
    """
    Split the dataframe to train/test/val by config.split_ratio.
    Each will be further processed if dataset balanced.

    :return: 3 dataframes for train, test and val.
    """

    def _retrieve_samples(df0: DataFrame, split_dict: dict):
        all_cls_samples = []
        for label, count in split_dict.items():
            samples = df0.loc[df0[config.label_field] == label]
            samples = samples.sample(count)
            all_cls_samples.append(samples)
            df0 = df0.drop(samples.index)
        return df0, pd.concat(all_cls_samples)

    count_dict = {i: len(df.loc[df[config.label_field] == i])
                  for i in range(label_features.num_classes)}
    train_dict, test_dict, val_dict = set_imb_count_dict(
        count_dict, config.imb_tolerance, config.imb_threshold,
        config.cls_ratio_to_imb, config.sample_ratio_to_imb,
        config.balance_strategy)

    df, train_df = _retrieve_samples(df, train_dict)
    df, test_df = _retrieve_samples(df, test_dict)
    _, val_df = _retrieve_samples(df, val_dict)

    return train_df, test_df, val_df


def main(config: DataConfig):
    ds = load_dataset(*config.huggingface_dataset_name)

    df = [ds[key].to_pandas() for key in ds]
    df = pd.concat(df)
    df.index = range(len(df))

    label_features = ds['train'].features[config.label_field]

    # Create ClassLabel object for the label field if it's not of the type.
    # Replace label in df from string to int.
    if type(label_features) is not ClassLabel:
        if "float" in label_features.dtype:
            df.loc[(df[config.label_field] >= 0.5), 'label'] = 1
            df.loc[(df[config.label_field] < 0.5), 'label'] = 0
            df[config.label_field] = df[config.label_field].astype("int32")

        label_features = ClassLabel(names=list(set(df[config.label_field].values)))
        replace_dict = {name: label_features.names.index(name) for name in label_features.names}
        df = df.replace(replace_dict)

    train, test, val = split_df(df, label_features, config)
    train, test, val = train[:config.test], test[:config.test], val[:config.test]
    train, test, val = TaskDataset(train, label_features, config), \
                       TaskDataset(test, label_features, config), \
                       TaskDataset(val, label_features, config),

    return train, test, val
