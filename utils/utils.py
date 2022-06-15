from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import json

BASE_RESULTS_PATH = './results'

def get_run_id(results_dir):
    run_id = 0
    results_dir = Path(BASE_RESULTS_PATH) / results_dir
    if results_dir.exists():
        id_list = [int(str(x).split("_")[-1]) for x in results_dir.iterdir()]
        run_id = 0 if not id_list else max(id_list) + 1
    return run_id


def save_results(args, linear_results, nmc_results, embeddings, scenario_id, wandb, run_id):
    if args.wandb:
        if linear_results is not None:
            wandb.run.summary[f"linear_results_seq_{scenario_id}"] = linear_results
        if nmc_results is not None:
            wandb.run.summary[f"nmc_results_seq_{scenario_id}"] = nmc_results

    results_dir = Path(BASE_RESULTS_PATH) / args.results_dir
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    run_dir = results_dir / f'run_{str(run_id).zfill(3)}'
    if not run_dir.exists():
        run_dir.mkdir()
        with open(run_dir / 'config.txt', 'w') as config:
            json.dump(vars(args), config)
    if linear_results is not None:
        linear_df = pd.DataFrame(linear_results)
        linear_df.to_csv(run_dir / f'case_{scenario_id}.acc.lin', float_format='%.3f')
    if nmc_results is not None:
        nmc_df = pd.DataFrame(nmc_results)
        nmc_df.to_csv(run_dir / f'case_{scenario_id}.acc.nmc', float_format='%.3f')
    with open(run_dir / f'case_{scenario_id}.emb', 'wb') as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
