source ~/.local/share/virtualenvs/imbtxtcls-EkdIcc14/bin/activate

python test_run_task.py -i 0 --model bert --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model gpt2 --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model xlnet --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model cnn --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model lstm --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model lstmattn --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model rcnn --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model mlp --tokenizer bert --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200
python test_run_task.py -i 0 --model han --tokenizer bert-sent --tokenizer_pretrained bert-base-uncased --embedder bert --test 0 --epoch 200

