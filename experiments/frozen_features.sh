RESULTS_DIR="frozen_features"
BASE_ARGS="--nmc --freeze-features --skip-eval --wandb --num-permutations 5 --lr 0.001 --optim sgd --num-epochs 30"
models=("resnet" "densenet" "vgg")
for model in ${models[@]}; do
    python main.py $BASE_ARGS --dataset cifar-10 --increment 2 --model $model --results-dir $RESULTS_DIR/$model

    python main.py $BASE_ARGS --dataset cifar-100 --increment 10 --model $model --results-dir $RESULTS_DIR/$model

#    python main.py $BASE_ARGS --dataset mnist --increment 2 --model $model --results-dir $RESULTS_DIR/$model

    python main.py $BASE_ARGS --dataset cub200 --increment 20 --model $model --results-dir $RESULTS_DIR/$model
done
