export LD_LIBRARY_PATH=/share/lvegna/anaconda3/envs/hiatus_scratch/lib/


python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" \
--model lstm --loss infoNCE --cross_genre --batch_size 256 --vcross_genre --vbatch_size 64 --scheduler enable \
--init_learning_rate 5e-3 \
--do_unfreeze --unfrozen_learning_rate 5e-4 --unfreeze_step_interval 500 --unfreeze_layer_limit 1 --unfreeze_direction -1 --warmup_steps 200 \
--training_steps 100000 \
--checkpoint '/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-01-22_16-46-10.ckpt'