import wandb

# Metric to optimize
monitor="Average Success at 8"

# Wether to maximize of minimize the monitored metric
monitor_mode="max"
assert(monitor_mode in ["max", "min"])

sweep_config = {
    'program': 'run_experiment.py',
    'method': 'bayes',
    'metric': {
        'name': monitor,
        'goal': 'maximize' if monitor_mode == 'max' else "minimize"
    },
    'parameters': {
        'output_projection_size': {
           'values': [4, 8, 16]
        }
    },
    'command': [
        '${env}',
        'python',
        '${program}',
        '--transformer', 'microsoft/deberta-v3-large',
        '--model_max_length', '512',
        '--model', 'projection_lstm',
        '--loss', 'infoNCE',
        '--blogs_caps', 
        '--vbatch_size', '64',
        '--batch_size', '456',
        '--valid_step_interval', '125',
        '--scheduler', 'enable',
        '--warmup_steps', '200',
        '--training_steps', '2250',
        '--wandb_disable',
        '--monitor', monitor,
        '--monitor_mode', monitor_mode,
        '${args}'
    ]
}


import os, sys

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

entity="author-edu"
project="PART-BaseSweep"

with SuppressPrints():
    sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)

print(f"{entity}/{project}/{sweep_id}")
