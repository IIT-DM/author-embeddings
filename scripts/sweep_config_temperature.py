import wandb

# Metric to optimize
monitor="Average Success at 8"

# Wether to maximize of minimize the monitored metric
monitor_mode="max"
assert(monitor_mode in ["max", "min"])

sweep_config = {
    'name' : "init_temperature_sweep",
    'program': 'run_experiment.py',
    'method': 'bayes',
    'metric': {
        'name': monitor,
        'goal': 'maximize' if monitor_mode == 'max' else "minimize"
    },
    'parameters': {
        'init_temperature': {
            'values': [.0002, .07, .2, 1, 2, 8, 64, 256, 512]
        },
        # 'dropout' : {
        #     'min': 0.0,
        #     'max': 1.0
        # }
    },
    'command': [
        '${env}',
        'python',
        '${program}',
        '--transformer', 'microsoft/deberta-v3-large',
        '--model_max_length', '512',
        '--model', 'lstm',
        '--loss', 'infoNCE',
        '--blogs_caps', 
        '--vbatch_size', '64',
        '--batch_size', '512',
        '--valid_step_interval', '30',
        '--scheduler', 'enable',
        '--training_steps', '500',
        '--wandb_disable',
        '--monitor', monitor,
        '--monitor_mode', monitor_mode,
        '--wandb_project', 'na'
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
