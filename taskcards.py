import torch
import vars

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from losses import FocalLoss, DiceLoss, TverskyLoss


def resample_9_ds(dc: dict, args):
    taskcards = []
    models = list(vars.model_names)
    loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss

    dc["balance_strategy"] = args.balance_strategy
    dc["make_it_imbalanced"] = args.make_it_imbalanced

    for model in models:
        mc = {
            "model_name": model,
            "pretrained_tokenizer_name": args.tokenizer_pretrained,
            "pretrained_model_name": args.model_pretrained,
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
            "early_stop_epoch": args.early_stop_epoch,
            "random_seed": args.random_seed,
        }

        taskcards.append(tc)

    return taskcards


def scenario_0(dc: dict, args):
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
            "early_stop_epoch": args.early_stop_epoch,
            "random_seed": args.random_seed,
            "retrain": args.retrain,
        }

        task_cards.append(tc)

    return task_cards


def scenario_1(args):
    dcs = [vars.datasets_meta[args.dataset_i]] if args.dataset_i != None else vars.datasets_meta[:13]
    task_cards = []

    models = list(vars.model_names) if not args.model else [args.model]
    for dc in dcs:
        dc["balance_strategy"] = args.balance_strategy
        dc["make_it_imbalanced"] = args.make_it_imbalanced

        loss_funcs = [BCEWithLogitsLoss] if dc.get('multi_label') else [CrossEntropyLoss]

        if args.loss == "all":
            loss_funcs = [FocalLoss, DiceLoss, TverskyLoss] + loss_funcs
        elif args.loss == "focal":
            loss_funcs = [FocalLoss]
        elif args.loss == "dice":
            loss_funcs = [DiceLoss]
        elif args.loss == "tversky":
            loss_funcs = [TverskyLoss]


        for model in models:
            for loss in loss_funcs:
                mc = {
                    "model_name": model,
                    "pretrained_tokenizer_name": args.tokenizer_pretrained,
                    "qkv_size": args.qkv_size,
                    "n_layers": args.layers,
                    "n_heads": args.n_heads,
                    "pretrained_model_name": args.model_pretrained,
                }
                tc = {
                    "data_config": dc,
                    "model_config": mc,
                    "batch_size": args.batch_size,
                    "loss_func": loss(),
                    "device": args.device,
                    "optimizer": torch.optim.AdamW,
                    "test": args.test if args.test else None,
                    "epoch": args.epoch,
                    "early_stop_epoch": args.early_stop_epoch,
                    "retrain": args.retrain,
                    "random_seed": args.random_seed,
                }

                task_cards.append(tc)

    return task_cards


def scenario_2(dc: dict, args):
    task_cards = []

    loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss
    models = zip(vars.transformer_names, vars.transformer_pretrain) if not \
        (args.model_pretrained and args.model) else [(args.model, args.model_pretrained)]
    for model, pretrained_name in models:

        mc = {
            "model_name": model,
            "pretrained_model_name": pretrained_name,
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
            "early_stop_epoch": args.early_stop_epoch,
            "random_seed": args.random_seed,
            "retrain": args.retrain,
        }

        task_cards.append(tc)

    return task_cards


def retrain(dc, args):
    dcs = vars.datasets_meta[:20] if not dc else [dc]
    task_cards = []

    models = ["gpt2", "lstmattn", "cnn", "rcnn"]

    for i, dc in enumerate(dcs):
        for model in models:
            loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss
            mc = {
                "model_name": model,
                "pretrained_tokenizer_name": "gpt2",
                "n_layers": args.layers,
                "disable_output": True,
                "disable_intermediate": True,
                "disable_selfoutput": True,
                "add_pooling_layer": False,
                "qkv_size": 768,
            }

            tc = {
                "data_config": dc,
                "model_config": mc,
                "batch_size": 20,
                "loss_func": loss(),
                "device": args.device,
                "optimizer": torch.optim.AdamW,
                "test": args.test if args.test else None,
                "epoch": args.epoch,
                "early_stop_epoch": args.early_stop_epoch,
                "retrain": args.retrain,
                "random_seed": args.random_seed,
            }

            task_cards.append(tc)

    return task_cards
