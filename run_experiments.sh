BASE_ARGS="--nmc --freeze-features --skip-eval --save-results"

python main.py $BASE_ARGS --dataset cifar-10 --increment 2 --num-permutations 1

# python main.py $BASE_ARGS --dataset cifar-100 --increment 10 --num-permutations 1

# python main.py $BASE_ARGS --dataset mnist --increment 2 --num-permutations 1

# python main.py $BASE_ARGS --dataset cub200 --increment 10 --num-permutations 1