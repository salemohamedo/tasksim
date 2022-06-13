BASE_ARGS="--num-epochs 10 --multihead --skip-eval --save-results --wandb --num-permutations 10 --lr 0.001 --optim sgd"

python main.py $BASE_ARGS --dataset cifar-10 --increment 2

python main.py $BASE_ARGS --dataset cifar-100 --increment 10

python main.py $BASE_ARGS --dataset mnist --increment 2

python main.py $BASE_ARGS --dataset cub200 --increment 20