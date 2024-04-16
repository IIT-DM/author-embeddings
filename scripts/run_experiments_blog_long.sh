export LD_LIBRARY_PATH=/share/lvegna/anaconda3/envs/hiatus_scratch/lib/




# python3 run_experiment.py  --transformer "google/bigbird-roberta-base" --model_max_length 4096 \
# --model lstm --loss infoNCE --blogs_BigBird  --batch_size 160  --vbatch_size 64 --scheduler enable --init_learning_rate 5e-4 \
# --training_steps 5000 --valid_step_interval 50 --no_iarpa --monitor "valid/infonce_acc"


# python3 run_experiment.py  --transformer "TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ" --model_max_length 2048 \
# --model lstm --loss infoNCE --blogs --num_to_multisample 1 4 --batch_size 100  --vbatch_size 10 --scheduler enable --init_learning_rate 5e-4 \
# --training_steps 5000 --valid_step_interval 125 


# python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" --model_max_length 9223372036854775808 --chunk_length 512  \
# --model long_lstm --loss infoNCE --blogs_caps --num_to_multisample 2 12 --batch_size 250  --vbatch_size 64 --scheduler enable \
# --training_steps 5000 --valid_step_interval 125

python3 run_experiment.py  --transformer "microsoft/deberta-v3-large" --model_max_length 8129 --chunk_length 512  \
--model long_attention --loss infoNCE --blogs_caps --num_to_multisample 2 16 --batch_size 350 --vbatch_size 64 --scheduler enable --init_temperature 4.0 \
--training_steps 4500 --valid_step_interval 50 --wandb_project "PART-Long" --monitor "Area Under ROC Curve"

