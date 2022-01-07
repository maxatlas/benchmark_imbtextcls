"""


"""
from datasets.features.features import ClassLabel
from datasets import load_dataset
from Config import DataConfig

from pandas.core.frame import DataFrame
from TaskDataset import set_imb_count_dict

import pandas as pd


class TaskDataset:
    def __init__(self, data: DataFrame, label_feature: ClassLabel, config: DataConfig):
        self.info = config
        self.data = data
        self.label_feature = label_feature

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]

        label = row[self.info.label_field]
        data = row.loc[self.info.text_fields]
        data = "\n".join(data)

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
        config.cls_ratio_to_imb, config.sample_ratio_to_imb)

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
        label_features = ClassLabel(names=list(set(df[config.label_field].values)))
        replace_dict = {name: label_features.names.index(name) for name in label_features.names}
        df = df.replace(replace_dict)

    train, test, val = split_df(df, label_features, config)
    # train, test, val = TaskDataset(train, label_features, config), \
    #                    TaskDataset(test, label_features, config), \
    #                    TaskDataset(val, label_features, config),

    return train, test, val


if __name__ == "__main__":
    from Config import DataConfig
    from task_config import datasets_meta
    config = DataConfig(*datasets_meta[-1].values())
    a, b, c = main(config)