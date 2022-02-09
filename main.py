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
                        required=True,
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

    args = parser.parse_args()
    dc = vars.datasets_meta[args.dataset_i]

    scene = taskcards.scenario_1
    if args.scenario == 2:
        scene = taskcards.scenario_2
    elif args.scenario == 3:
        scene = taskcards.scenario_3
    elif args.scenario == 0:
        scene = taskcards.scenario_0

    tasks = scene(dc, args)
    for task in tasks:

        model_path = None
        if args.retrain:
            model_path = vars.trained_model_cur_folder + "/" +\
                         task["model_config"]["model_name"]
        try:
            run_task.main(TaskConfig(**task))
        except Exception as e:
            print(e)
