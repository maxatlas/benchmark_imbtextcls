import torch
import vars

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from losses import FocalLoss, DiceLoss, TverskyLoss


def resample_9_ds(dc: dict, args):
    taskcards = []
    models = list(vars.model_names)
    loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss

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
            "balance_strategy": args.balance_strategy,
            "make_it_imbalanced": args.make_it_imbalanced,
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
        }

        task_cards.append(tc)

    return task_cards


def scenario_1(dc: dict, args):
    task_cards = []
    loss_funcs = [FocalLoss, DiceLoss, TverskyLoss]
    # loss = BCEWithLogitsLoss if dc.get('multi_label') else CrossEntropyLoss
    models = list(vars.model_names) if not args.model else [args.model]
    for model in models:
        for loss in loss_funcs:
            print(model, loss)
            mc = {
                "model_name": model,
                "pretrained_tokenizer_name": args.tokenizer_pretrained,
                "qkv_size": 768,
                "n_layers": args.layers,
                "n_heads": 1,
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
            }

            task_cards.append(tc)

    return task_cards
