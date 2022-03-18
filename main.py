"""
check max word length and sent length, configure model word max length accordingly properly
"""
import vars
import run_task as run_task
import argparse
import taskcards
from Config import TaskConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_i',
                        '-i',
                        default=None,
                        required=False,
                        type=int,
                        help="nth dataset from vars.datasets_meta")
    parser.add_argument("--device",
                        default="cuda:0",
                        type=str,
                        help="which device to run on: cpu/cuda:n")
    parser.add_argument('--tokenizer_pretrained',
                        '-x',
                        type=str,
                        default="")
    parser.add_argument('--model_pretrained',
                        type=str,
                        default="")
    parser.add_argument("--test",
                        '-t',
                        default=0,
                        type=int)
    parser.add_argument('--epoch',
                        '-e',
                        type=int,
                        default=3)
    parser.add_argument('--layers',
                        '-l',
                        type=int,
                        default=1)
    parser.add_argument("--early_stop_epoch",
                        '-s',
                        type=int,
                        default=5)
    parser.add_argument("--scenario",
                        type=int,
                        default=1)
    parser.add_argument("--retrain",
                        type=int,
                        default=0)
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        default=None)
    parser.add_argument("--batch_size",
                        type=int,
                        default=100
                        )
    parser.add_argument("--balance_strategy",
                        type=str,
                        default="")
    parser.add_argument("--make_it_imbalanced",
                        type=bool,
                        default=True)

    args = parser.parse_args()
    print(args.dataset_i)
    dc = vars.datasets_meta[args.dataset_i] if args.dataset_i != None else None
    dc['balance_strategy'] = args.balance_strategy
    dc['make_it_imbalanced'] = args.make_it_imbalanced

    scene = taskcards.scenario_1
    if args.scenario == 2:
        scene = taskcards.scenario_2
    elif args.scenario == 0:
        scene = taskcards.scenario_0
    elif args.scenario == 4:
        scene = taskcards.resample_9_ds

    if args.retrain:
        scene = taskcards.retrain
    tasks = scene(dc, args)
    for task in tasks:

        model_path = None
        if args.retrain:
            model_path = vars.trained_model_cur_folder + "/" +\
                          "%s_layer_%i" % (task["model_config"]["model_name"],
                                           task["model_config"]["n_layers"])
        # try:
        run_task.main(TaskConfig(**task), model_path=model_path)
        # except Exception as e:
        #     print(e)
