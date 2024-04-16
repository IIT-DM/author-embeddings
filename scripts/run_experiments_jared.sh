# export LD_LIBRARY_PATH=/share/lvegna/anaconda3/envs/hiatus_scratch/lib/


python3 run_experiment.py  --transformer "prajjwal1/bert-tiny" --model_max_length 512 \
--model lstm --loss infoNCE --hrs --batch_size 8 --vbatch_size 64 --scheduler enable \
--training_steps 5 --valid_step_interval 2 --wandb_project "PART-Extension"


