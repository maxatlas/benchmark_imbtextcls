import copy
import os
import vars
import dill
import json
import torch

from hashlib import sha256
from pathlib import Path
from datasets import load_dataset


def rename_gpt2_res(folder):
    def fix_obj(results, new_id=None):
        results_copy = copy.deepcopy(results)
        idx_dict = {}
        for idx, res in results_copy.items():
            mc = res['task']['model_config']
            mc['disable_selfoutput'] = False
            new_idx = sha256(str(res).encode('utf-8')).hexdigest() \
                if not new_id else new_id
            results[new_idx] = res
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
                    if f.endswith(".json"):
                        json.dump(results, open(file_name, "w"))
                    else:
                        results = dill.load(open(file_name, "rb"))
                        results_copy = copy.deepcopy(results)

                        for idx, roc_list in results_copy.items():
                            if idx_dict.get(idx):
                                results[idx_dict[idx]] = roc_list
                        dill.dump(results, open(file_name, "wb"))


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


def merge_res_from_sources(folders, destination):
    os.makedirs(destination, exist_ok=True)
    for data in vars.dataset_names:
        data = data + "_balance_strategy_None"
        for model in vars.model_names:
            print(data, model)
            file = Path(data, model)
            jfile = Path(data, model+".json")
            results = {}
            for folder in folders:
                try:
                    res = dill.load(open(folder/file, "rb"))
                    for idx, value in res.items():
                        if idx in results:
                            results[idx]["result"].extend(value["result"])
                        else:
                            results.update({idx: value})
                except FileNotFoundError:
                    continue
                except EOFError:
                    continue
            if results:
                os.makedirs(destination+"/%s" % data, exist_ok=True)
                dill.dump(results, open(destination/file, "wb"))
                json.dump(results, open(destination/jfile, "w"))


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

    return overall_length


def get_ds_lengths(dmetas=None):
    if not dmetas:
        dmetas = vars.datasets_meta

    for dmeta in dmetas:
        print(dmeta['huggingface_dataset_name'])
        out = get_ds_length(dmeta)
        out.sort()
        print("avg length: %f" % (sum(out)/len(out)))
        print("median length: %i" % out[int(len(out)/2)])


def get_model_param_size(model):
    out = 0
    for i, layer in enumerate(model.parameters()):
        if layer.requires_grad:
            out += torch.prod(torch.tensor(layer.size()))
    return out


if __name__ == "__main__":
    rename_gpt2_res("results")
    # path = "trained/bert/9353c651b23d3a4aca18a8b34480ffa6cdd0e9c2761097ab925e2e9a6a00f8c9"
    # model = torch.load(path)
    # size = get_model_param_size(model)
    # print(size)
    # # merge_res_from_sources(["res_uq", "results"], "merged")
