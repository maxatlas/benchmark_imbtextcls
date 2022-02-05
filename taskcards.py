import torch
import vars

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss


def scenario_1(dc: dict, args):
    task_cards = []
    loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss
    models = list(vars.model_names)
    for model in models:

        mc = {
            "model_name": model,
            "pretrained_tokenizer_name": args.tokenizer_pretrained,
            "n_layers": args.layers,
        }
        tc = {
            "data_config": dc,
            "model_config": mc,
            "batch_size": 100,
            "loss_func": loss(),
            "device": args.device,
            "optimizer": torch.optim.AdamW,
            "test": args.test if args.test else None,
            "epoch": args.epoch,
            "early_stop_alpha": args.early_stop_alpha,
        }

        task_cards.append(tc)

    return task_cards


def scenario_2(dc: dict, args):
    task_cards = []

    loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss
    for model, pretrained_name in zip(vars.transformer_names, vars.transformer_pretrain):

        mc = {
            "model_name": model,
            "pretrained_model_name": pretrained_name
        }

        tc = {
            "data_config": dc,
            "model_config": mc,
            "batch_size": 100,
            "loss_func": loss(),
            "device": args.device,
            "optimizer": torch.optim.AdamW,
            "test": args.test if args.test else None,
            "epoch": args.epoch,
            "early_stop_alpha": args.early_stop_alpha,
        }

        task_cards.append(tc)

    return task_cards


def scenario_3(dc: dict, args):
    task_cards = []

    loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss
    for model, pretrained_name in zip(vars.transformer_names, vars.transformer_pretrain):
        mc = {
            "model_name": model,
            "pretrained_tokenizer_name": pretrained_name,
            "n_layers": args.layers,
        }

        tc = {
            "data_config": dc,
            "model_config": mc,
            "batch_size": 100,
            "loss_func": loss(),
            "device": args.device,
            "optimizer": torch.optim.AdamW,
            "test": args.test if args.test else None,
            "epoch": args.epoch,
            "early_stop_alpha": args.early_stop_alpha,
        }

        task_cards.append(tc)

    return task_cards
