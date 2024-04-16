PYTHONPATH=/share/lvegna/Repos/author/authorship-embeddings \
CUDA_VISIBLE_DEVICES=0 \
python3 /share/lvegna/Repos/author/authorship-embeddings/author-ml/contrastive_learning.py \
decoding \
--model_checkpoint /share/lvegna/Repos/author/authorship-embeddings/model/final_2023-09-09_09-54-01_lstm_blogs_electra-large-discriminator_infoNCE.ckpt \
--distance euclidean