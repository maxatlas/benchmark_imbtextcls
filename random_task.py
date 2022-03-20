import copy
import os

import pandas.core.frame

import vars
import dill
import json
import pandas as pd
from build_dataset import split_df, TaskDataset
from hashlib import sha256
from pathlib import Path
from datasets import load_dataset
from Config import DataConfig

from dataset_utils import get_imb_factor, get_count_dict


def exclude_index_for_cache(i: int, filename: str):
    dfcur = dill.load(open(".cache/exp/%s" % filename, "rb"))[0]
    train, test, val, split_info = df_exclude_index(i, dfcur)
    assert dfcur.data.index[0] not in train.data.index
    dill.dump((train, test, val, split_info), open(
        ".cache/exp/dataset/%s" % filename,
        "wb"))


def df_exclude_index(i: int, df_to_exclude: pandas.core.frame.DataFrame):
    meta = vars.datasets_meta[i]
    meta['balance_strategy'] = "uniform"
    ds = load_dataset(*meta['huggingface_dataset_name'])
    df = [ds['train'].to_pandas(), ds['test'].to_pandas()]
    df = pd.concat(df)
    bad_index = df.index.isin(df_to_exclude.data.index)
    df = df[~bad_index]
    label_feature = ds['train'].features[meta['label_field']]
    dc = DataConfig(**meta)
    train, test, val, split_info = split_df(df, label_feature, dc)
    train = TaskDataset(train, label_feature, dc)
    test = TaskDataset(test, label_feature, dc)
    val = TaskDataset(val, label_feature, dc)
    return train, test, val, split_info


def delete_XLNET(folder):
    for ds in os.listdir(folder):
        res_folder = vars.results_folder + "/%s" % ds
        for f in os.listdir(res_folder):
            if f.startswith("xlnet"):
                os.remove(Path(folder, ds, f))


def rename_gpt2_res(folder):
    def fix_obj(results, new_id=None):
        results_copy = copy.deepcopy(results)
        idx_dict = {}
        for idx, res in results_copy.items():
            mc = res['task']['model_config']
            if mc.get('disable_selfoutput') is None:
                mc['disable_selfoutput'] = False
                mc['disable_output'] = True
                mc['disable_intermediate'] = True
                new_idx = sha256(str(res).encode('utf-8')).hexdigest() \
                    if not new_id else new_id
                results[new_idx] = res
                del results[idx]
                idx_dict[idx] = new_idx
                print(results[new_idx]['task']['model_config'])
        return results, idx_dict

    for ds in os.listdir(folder):
        res_folder = vars.results_folder + "/%s" % ds
        for f in os.listdir(res_folder):
            if f.startswith("gpt2") or f.startswith("xlnet"):
                file_name = res_folder + "/" + f
                print("\n\n"+file_name)
                if f.endswith("xlnet") or f.endswith("gpt2"):
                    results = dill.load(open(file_name, "rb"))
                    results, idx_dict = fix_obj(results)
                    dill.dump(results, open(file_name, "wb"))
                else:
                    if f.endswith(".roc"):
                        results = dill.load(open(file_name, "rb"))
                        results_copy = copy.deepcopy(results)

                        for idx, roc_list in results_copy.items():
                            if idx_dict.get(idx):
                                results[idx_dict[idx]] = roc_list
                                del results[idx]
                        dill.dump(results, open(file_name, "wb"))


def remove_oversample(folder):
    for ds in os.listdir(folder):
        res_folder = vars.results_folder + "/%s" % ds

        for model in vars.model_names:
            filename = res_folder+"/%s" % model
            print(filename)
            try:
                res = dill.load(open(filename, "rb"))
                roc = dill.load(open(filename+".roc", "rb"))

                res_examine = copy.deepcopy(res)
                for key, value in res_examine.items():
                    if value['task']['data_config']['balance_strategy'] == "oversample":
                        print("Key %s to be deleted from dill file." %key)
                        del res[key]
                        try:
                            del roc[key]
                        except Exception:
                            print("Key %s not in .roc file" %key)
                print(res.keys())
                dill.dump(res, open(filename+"", 'wb'))
                dill.dump(roc, open(filename+"" + ".roc", "wb"))
            except FileNotFoundError:
                continue


def rename_cnn_res(folder):
    def fix_obj(results, new_id=None):
        results_copy = copy.deepcopy(results)
        new_res, new_idx = None, None
        for idx, res in results_copy.items():
            mc = res['task']['model_config']
            if mc['num_layers'] == 1:
                new_res = copy.deepcopy(res)
                new_res['task']['model_config']['num_layers'] = 4
                new_idx = sha256(str(new_res).encode('utf-8')).hexdigest() \
                    if not new_id else new_id
                new_res = {new_idx: new_res}
                if idx != new_idx:
                    print("update!")
                    print(new_res[new_idx]['task']['model_config'])
                    results.update(new_res)
        return results, new_res, new_idx, idx

    for ds in os.listdir(folder):
        res_folder = vars.results_folder + "/%s" % ds
        new_res, idx, new_idx = {}, None, None
        for f in os.listdir(res_folder):
            if f.startswith("cnn"):
                file_name = res_folder + "/" + f
                print("\n\n"+file_name)
                if f.endswith("cnn"):
                    results = dill.load(open(file_name, "rb"))
                    results, new_res, new_idx, idx = fix_obj(results)
                    dill.dump(results, open(file_name, "wb"))
                else:
                    if f.endswith(".json"):
                        results = json.load(open(file_name))
                        results, new_res, new_idx, idx = fix_obj(results, new_id=new_idx)
                        json.dump(results, open(file_name, "w"))
                    else:
                        results = dill.load(open(file_name, "rb"))
                        obj = results[idx]
                        new_res = {new_idx: obj}
                        results.update(new_res)
                        dill.dump(results, open(file_name, "wb"))

                print(list(results.keys()))


def get_ds_length(dmeta):
    data_name = dmeta["huggingface_dataset_name"]
    text_fields = dmeta["text_fields"]
    ds = load_dataset(*data_name)

    overall_length = []
    for split in ds.values():
        for sample in split:
            for text_field in text_fields:
                text = sample[text_field]
                overall_length.append(len(text))
    overall_length.sort()
    df = pd.DataFrame(overall_length)

    return df


def get_ds_lengths(dmetas=None):
    if not dmetas:
        dmetas = vars.datasets_meta

    for dmeta in dmetas:
        print(dmeta['huggingface_dataset_name'])
        df = get_ds_length(dmeta)
        print(df.quantile([0.25, 0.5, 0.75]))


from datasets.features.features import ClassLabel, Value, Sequence
def get_ds_if(dmeta):
    data_name = dmeta["huggingface_dataset_name"]
    ds = load_dataset(*data_name)
    label_field = dmeta['label_field']
    df = [ds[key].to_pandas() for key in ds]
    df = pd.concat(df)

    label_features = ds['train'].features[dmeta['label_field']]
    if type(label_features) is Value:
        if "float" in label_features.dtype:
            df.loc[(df[dmeta['label_field']] >= 0.5), 'label'] = 1
            df.loc[(df[dmeta['label_field']] < 0.5), 'label'] = 0
            df[dmeta['label_field']] = df[dmeta['label_field']].astype("int32")

    count_dict = get_count_dict(df[label_field])
    return get_imb_factor(count_dict)


def get_ds_ifs(dmetas=None):
    if not dmetas:
        dmetas = vars.datasets_meta

    for dmeta in dmetas:
        print(dmeta['huggingface_dataset_name'])
        imb_factor = get_ds_if(dmeta)

        print(imb_factor)


if __name__ == "__main__":
    remove_oversample(vars.results_folder)