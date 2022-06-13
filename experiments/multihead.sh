RESULTS_DIR="multihead"
BASE_ARGS="--num-epochs 10 --multihead --skip-eval --wandb --num-permutations 10 --lr 0.001 --optim sgd"
models=("resnet" "densenet" "vgg")
for model in ${models[@]}; do
    python main.py $BASE_ARGS --dataset cifar-10 --increment 2 --model $model --results-dir $RESULTS_DIR/$model

    python main.py $BASE_ARGS --dataset cifar-100 --increment 10 --model $model --results-dir $RESULTS_DIR/$model

#    python main.py $BASE_ARGS --dataset mnist --increment 2 --model $model --results-dir $RESULTS_DIR/$model

    python main.py $BASE_ARGS --dataset cub200 --increment 20 --model $model --results-dir $RESULTS_DIR/$model
done