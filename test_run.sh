#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=max_benchmark0-13
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=300000
#SBATCH -o bench_out.txt
#SBATCH -e bench_error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/10.0.130

source benchmark_imbtextcls/bin/activate


export HF_DATASETS_CACHE="/scratch/itee/uqclyu1/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/scratch/itee/uqclyu1/.cache/modules"

srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --model bert -p bert-base-uncased
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --model gpt2 -p gpt2
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --model xlnet -p xlnet-base-cased
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --model roberta -p roberta-base
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model cnn -b glove
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model cnn --tokenizer_pretrained bert-base-uncased
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model rcnn -b glove
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model rcnn --tokenizer_pretrained bert-base-uncased
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model lstm -b glove
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model lstm --tokenizer_pretrained bert-base-uncased
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model lstmattn -b glove
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model lstmattn --tokenizer_pretrained bert-base-uncased
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model mlp -b glove
srun python benchmark_imbtextcls/test_run_task.py --epoch 5 --test 0 --dataset_i 11 --tokenizer spacy --model mlp --tokenizer_pretrained bert-base-uncased