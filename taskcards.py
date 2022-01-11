import torch
import vars

from torch.nn import BCEWithLogitsLoss


def scenario_1(dc, args):
    task_cards = []
    for model, pretrain in list(zip(vars.transformer_names,
                                    vars.transformer_pretrain)):
        mc = {
            "model_name": model,
            "pretrained_model_name": pretrain,
        }
        tc = {
            "data_config": dc,
            "model_config": mc,
            "batch_size": 100,
            "loss_func": BCEWithLogitsLoss(),
            "device": args.device,
            "optimizer": torch.optim.AdamW,
            "test": 3 if args.test else None,
        }

        task_cards.append(tc)

    return task_cards


def scenario_2(dc, args):
    task_cards = []

    for model in vars.model_names[4:-1]:
        for pretrain in vars.transformer_pretrain:
            mc = {
                "model_name": model,
                "pretrained_tokenizer_name": pretrain,
            }
            tc = {
                "data_config": dc,
                "model_config": mc,
                "batch_size": 100,
                "loss_func": BCEWithLogitsLoss(),
                "device": args.device,
                "optimizer": torch.optim.AdamW,
                "test": 3 if args.test else None,
            }
            task_cards.append(tc)

    for model, pretrain in list(zip(vars.transformer_names,
                                    vars.transformer_pretrain)):
        mc = {
            "model_name": model,
            "pretrained_tokenizer_name": pretrain,
        }
        tc = {
            "data_config": dc,
            "model_config": mc,
            "batch_size": 100,
            "loss_func": BCEWithLogitsLoss(),
            "device": args.device,
            "optimizer": torch.optim.AdamW,
            "test": 3 if args.test else None,
        }

        task_cards.append(tc)

    return task_cards


def scenario_3(dc, args):
    task_cards = []
    for model in vars.customized_model_names:
        for tok in vars.customized_tokenizer_names:
            for emb_path in ["glove", "fasttext", "word2vec"]:
                mc = {
                    "model_name": model,
                    "tokenizer_name": tok,
                    "emb_path": "%s/%s" % (vars.parameter_folder, emb_path)
                }
                tc = {
                    "data_config": dc,
                    "model_config": mc,
                    "batch_size": 100,
                    "loss_func": BCEWithLogitsLoss(),
                    "device": args.device,
                    "optimizer": torch.optim.AdamW,
                    "test": 3 if args.test else None,
                }
                task_cards.append(tc)

    return task_cards
