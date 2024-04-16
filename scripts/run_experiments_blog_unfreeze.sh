
export LD_LIBRARY_PATH=/share/lvegna/anaconda3/envs/hiatus_scratch/lib/


# python3 run_experiment.py  \
# --transformer "google/electra-large-discriminator" --model lstm \
# --blogs \
# --loss infoNCE \
# --wandb_disable \
# --training_steps 2000 --batch_size 2 --vbatch_size 100 --valid_step_interval 250 \
# --scheduler enable --init_learning_rate 5e-3 --warmup_steps 200 \
# --do_unfreeze --unfrozen_learning_rate 5e-4 --unfreeze_step_interval 500 --unfreeze_layer_limit 1 --unfreeze_direction -1 \
# # --checkpoint "model/final_2023-10-17_13-32-53_lstm_blogs_electra-large-discriminator_infoNCE_batsz128..ckpt"





python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" \
--model lstm --loss infoNCE --blogs_caps --batch_size 256 --vbatch_size 64 --scheduler enable \
--init_learning_rate 5e-3 \
--do_unfreeze --unfrozen_learning_rate 5e-4 --unfreeze_step_interval 1000 --unfreeze_layer_limit 1 --unfreeze_direction -1 --warmup_steps 200 \
--training_steps 10000 --valid_step_interval 125 
