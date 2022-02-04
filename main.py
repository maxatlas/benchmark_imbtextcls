"""
check max word length and sent length, configure model word max length accordingly properly
"""
import vars
import run_task
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
    parser.add_argument("--full_run",
                        '-f',
                        type=int,
                        default=0)

    args = parser.parse_args()
    dc = vars.datasets_meta[args.dataset_i]

    print("Pretrained models ...")

    tasks = taskcards.scenario_1(dc, args)

    for task in tasks:
        try:
            run_task.main(TaskConfig(**task))
        except Exception as e:
            print(e)
