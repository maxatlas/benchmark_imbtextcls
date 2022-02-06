import copy
import os
import vars
import dill
import json

from hashlib import sha256
from pathlib import Path

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


if __name__ == "__main__":
    merge_res_from_sources(["results", "results_backup"], "test")
