export LD_LIBRARY_PATH=/share/lvegna/anaconda3/envs/hiatus_scratch/lib/


python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" \
--model lstm --loss infoNCE --cross_genre --batch_size 232 \
--vcross_genre --vbatch_size 64 --no_iarpa --valid_step_interval 5 \
--init_learning_rate 5e-3 --scheduler enable --warmup_steps 200 \
--head_hidden_size 256 --head_input_size 64 --num_layers 1 --dropout 0.1 \
--training_steps 2500 --monitor "valid/infonce_acc" --wandb_project "PART-CrossGenre"
