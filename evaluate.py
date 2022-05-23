import argparse

from pathlib import Path
from scipy.stats import pearsonr
import pickle
import pandas as pd
import numpy as np
from similarity_metrics.task2vec import cosine
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

RESULTS_PATH = './results'

parser = argparse.ArgumentParser(description='Tasksim Evaluate')
parser.add_argument('--run-dir', required=True)

args = parser.parse_args()

results_dir = Path(RESULTS_PATH)
if not results_dir.exists():
    print("No Results Directory!")
run_dir = results_dir / args.run_dir
if not run_dir.exists():
    print(f"No Run Directory: {run_dir}")

id_list = [int(str(x).split("_")[-1].split(".")[0]) for x in run_dir.iterdir()]
num_cases = max(id_list) + 1
print(f"\n## Total Number Cases: {num_cases}")
error_rates = []
cos_diffs = []
for case_id in range(num_cases):
    results = pd.read_csv(run_dir / f"case_{case_id}.acc")
    results = results.to_numpy()
    mean_error_rate = 1 - results[-1][1:].mean()
    error_rates.append(mean_error_rate)

    with open(run_dir / f'case_{case_id}.emb', 'rb') as f:
        embeddings = pickle.load(f)
    total_distance = 0
    for i in range(len(embeddings) - 1):
        total_distance += cosine(embeddings[i], embeddings[i + 1])
    cos_diffs.append(total_distance)

plt.scatter(cos_diffs, error_rates, marker='o')
plt.xlabel("Total task dissimilarity")
plt.ylabel("Mean task error rate")
plt.savefig("test.png")

print(f"Pearson correlation coefficient: {pearsonr(error_rates, cos_diffs)[0]}")