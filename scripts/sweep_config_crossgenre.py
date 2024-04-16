import wandb

# Metric to optimize
monitor="valid/infonce_acc"

# Wether to maximize of minimize the monitored metric
monitor_mode="max"
assert("max" in ["max", "min"])

sweep_config = {
    'program': 'run_experiment.py',
    'method': 'bayes',
    'metric': {
        'name': monitor,
        'goal': 'maximize' if monitor_mode == 'max' else "minimize"
    },
    'parameters': {
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'init_learning_rate': {
            'min': 0.000001,
            'max': 0.01
        },
        'head_hidden_size': {
            'values': [32, 64, 128, 256]
        },
        'head_input_size': {
            'values': [ 64, 128, 256, 512]
        }
    },
    'command': [
        '${env}',
        'python',
        '${program}',
        '--transformer', 'microsoft/deberta-v3-large',
        '--model', 'lstm', 
        '--loss', 'infoNCE',
        '--cross_genre',
        '--vcross_genre',
        '--vbatch_size', '8',
        '--no_iarpa',
        '--valid_step_interval', '10',
        '--scheduler', 'enable',
        '--warmup_steps', '200',
        '--training_steps', '500',
        '--checkpoint', '/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-01-22_16-46-10.ckpt',
        '--wandb_disable',
        '--monitor', monitor,
        '--monitor_mode' , monitor_mode,
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
project="PART-Sweeps"
name = "CrossGenre"

with SuppressPrints():
    sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project, name=name)

print(f"{entity}/{project}/{sweep_id}")
