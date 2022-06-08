import argparse

from utils.eval_utils import get_run_results, process_run_results, plot_similarity_correlation, load_config

parser = argparse.ArgumentParser(description='Tasksim Evaluate')
parser.add_argument('--run-dir', required=True)
args = parser.parse_args()

run_config = load_config(args.run_dir)
run_results = get_run_results(args.run_dir)
summarized_results = process_run_results(*run_results)

print(run_config)
print(summarized_results)
