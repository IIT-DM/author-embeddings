from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/electra-large-discriminator")
tok.save_pretrained_to_buffer()