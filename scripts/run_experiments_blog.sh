export LD_LIBRARY_PATH=/share/lvegna/anaconda3/envs/hiatus_scratch/lib/



# # Best model settings
# python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" --model_max_length 512 \
# --model lstm --loss infoNCE --blogs_caps --batch_size 512  --vbatch_size 64 --scheduler enable \
# --training_steps 1000 --valid_step_interval 40 --wandb_project "PART-Extension" --init_temperature 10.0 \

 
# Projection with best model settings
python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" --model_max_length 512 \
--model lstm --loss infoNCE --blogs_caps --batch_size 512  --vbatch_size 64 --scheduler enable \
--training_steps 1000 --valid_step_interval 40 --wandb_project "PART-Extension" --init_temperature 10.0 \
sampled_encoder_layers


 