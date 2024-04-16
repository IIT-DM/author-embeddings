python3 run_experiment.py  --transformer "google/electra-large-discriminator" \
 --model lstm  --blogs --scheduler enable \
 --loss supcon --n_views 12 --batch_size 12  --vbatch_size 100 --valid_step_interval 100\
