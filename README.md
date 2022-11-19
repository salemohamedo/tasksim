# Continual Learning Task Similarity Experiments

1. Create virtual env
2. Install requirements.txt
3. Add tiny dataset to data/ folder. You can download the dataset [here](https://drive.google.com/file/d/189yK05T-fWmnTnewQxNzX2DGjwnLEjHM/view?usp=sharing). The other datasets will download automatically when invoked. 
4. Run!

Example:
```python main.py --model resnet50 --n-classes-per-task 2 --n-tasks 4 --dataset cifar-100 --metrics```

The ```--metrics``` command will evaluate all metrics implemented in ```metrics.py```. However, task2vec is implemented elsewhere and needs to be invoked with ```--task2vec```. 

See ```util/tasksim_args.py``` for available CLI args. Runs can also be invoked with wandb sweeps, see ```sweeps/pretrained/class_inc_all_pretrained.yml``` for an example. 

