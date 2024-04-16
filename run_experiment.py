###############################################################################
# Imports #####################################################################
###############################################################################
from model import *
from data import build_dataset, build_supervised_dataset, build_genre_sampling_dataset
import os
from tqdm import tqdm
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datetime import datetime
import pandas as pd
import numpy as np
import wandb
import torch
torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


###############################################################################
# Runtime parameters ##########################################################
###############################################################################
arg_parser = ArgumentParser(description='Run an experiment.')
arg_parser.add_argument('--books', action='store_true',
                        help='Use books dataset')
arg_parser.add_argument('--mails', action='store_true',
                        help='Use mails dataset')
arg_parser.add_argument('--blogs', action='store_true',
                        help='Use blogs dataset')
arg_parser.add_argument('--blogs_caps', action='store_true',
                        help='Use blogs dataset')
arg_parser.add_argument('--blogs_BigBird', action='store_true',
                        help='Use blogs BigBird dataset')
arg_parser.add_argument('--reddit', action='store_true',
                        help='Use reddit dataset')
arg_parser.add_argument('--hrs', action='store_true',
                        help='Use hrs dataset')
arg_parser.add_argument('--cross_genre', action='store_true',
                        help='Use cross-genre dataset in cross-genre only mode')
arg_parser.add_argument('--transformer_adapter', action='store_true',
                        help='Use Low Rank Adapters on the transformer')
arg_parser.add_argument('--wandb_disable', action='store_true',
                        help='disable wandb')
arg_parser.add_argument('--model', type=str, required=True, help='Model type',
                        choices=['lstm', "projection_lstm", "long_lstm", 'transformer', "long_attention"])
arg_parser.add_argument('--scheduler', type=str, default='none', help='Model type',
                        choices=['enable', 'none'],
                        )
arg_parser.add_argument('--transformer', type=str, default='roberta-large', help='Model type',
                        )
arg_parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
arg_parser.add_argument('--vcross_genre', action='store_true',
                        help='Use cross genre dataset for validation in cross genre only mode')
arg_parser.add_argument('--vbatch_size', type=int,
                        default=0, help='Validation batch size')
arg_parser.add_argument('--output_projection_size', type=int,
                        default=256, help='This directly controls the ouput dimension of the Projection_LSTM')
arg_parser.add_argument('--head_hidden_size', type=int,
                        default=128, help='Head hidden dimesion width, the final output dimension will be 2x this value for the LSTM')
arg_parser.add_argument('--head_input_size', type=int,
                        default=256, help='The size of the input dimension of the LSTM. Controls the linear projection layer output dimension')
arg_parser.add_argument('--loss', type=str, default='infoNCE', help='Loss function to use',
                        choices=['infoNCE', 'supcon', 'infoNCE_euclidean', "hard_negatives"])
arg_parser.add_argument('--n_views', type=int, default=16,
                        help='Num views in SupConLoss')
arg_parser.add_argument('--n_negatives', type=int,
                        default=16, help='Num negatives in H-SCL')
arg_parser.add_argument('--devices', type=int,
                        default=torch.cuda.device_count(), help='Devices to use')
arg_parser.add_argument('--checkpoint', type=str, required=False, help='Model checkpoint to start from',
                        default=None,
                        )
arg_parser.add_argument('--monitor', type=str, required=False, help='Metric to monitor for sweep/checkpointing',
                        default="Average Success at 8",
                        )
arg_parser.add_argument('--monitor_mode', type=str, required=False, choices=['max', 'min'], help='Wether to maximize or minimize the monitored metric',
                        default="max",
                        )
arg_parser.add_argument('--valid_step_interval', type=int,
                        default=250, help='')
arg_parser.add_argument('--no_iarpa', action='store_true',
                        help='dont do Iarpa5 eval')
arg_parser.add_argument('--model_max_length', type=int,
                        default=512, help='max input dimension/length of the model')
arg_parser.add_argument('--num_to_multisample', type=int, nargs="+",
                        default=[1], help='The number of samples to concatenate together from the same author, use one number (--num_to_multisample 8) for fixed and two for range (--num_to_multisample 2 12)')
arg_parser.add_argument('--sampled_encoder_layers', type=int, nargs="+",
                        default=[-1], help='The layers to sample from the encoder to pass to the lstm -1 will take last hidden state 4 -1 will take the mean of last hidden and the fourth hidden')
arg_parser.add_argument('--chunk_length', type=int,
                        default=None, help='Length to split the input sequence into for longLSTM, this should typically be less or equal to the transformers max_position_embeddings')
arg_parser.add_argument('--init_learning_rate', type=float,
                        default=5e-3, help='Initial learning rate used during regular training')
arg_parser.add_argument('--dropout', type=float,
                        default=0.1, help='Dropout rate')
arg_parser.add_argument('--num_layers', type=int,
                        default=1, help='Number of layers')
arg_parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay')
arg_parser.add_argument('--training_steps', type=int,
                        default=1000, help='Number of training steps')
arg_parser.add_argument('--warmup_steps', type=int,
                        default=200, help='Number of warmup steps')
arg_parser.add_argument('--do_unfreeze', action='store_true',
                        help='Whether to perform progressive unfreezing')
arg_parser.add_argument('--unfrozen_learning_rate',
                        type=float, default=5e-4, help='Unfrozen learning rate')
arg_parser.add_argument('--init_temperature',
                        type=float, default=3.0, help='Sets the initial temperature used when caluclating InfoNCE loss')
arg_parser.add_argument('--unfreeze_step_interval',
                        type=int, default=500, help='Unfreeze step interval')
arg_parser.add_argument('--unfreeze_layer_limit', type=int,
                        default=float('inf'), help='Maximum number of layers to unfreeze')
arg_parser.add_argument('--unfreeze_direction', type=int, default=-1,
                        choices=[-1, 1], help='Unfreeze direction: 1 for input to output, -1 for output to input')
arg_parser.add_argument('--valid_step_interval', type=int, default=100, help='Batch size')
# arg_parser.add_argument('--wandb_project', required=True, type=str, help="The wandb_project to use")
args = arg_parser.parse_args()

BATCH_SIZE = args.batch_size
HEAD_HIDDEN_SIZE = args.head_hidden_size
HEAD_INPUT_SIZE = args.head_input_size
PROJECTION_OUTPUT_SIZE = args.output_projection_size
VALID_BATCH_SIZE = args.vbatch_size
ENABLE_SCHEDULER = args.scheduler == 'enable'
DEVICES = args.devices
MODEL_TYPE = args.model
BASE_CODE = args.transformer
CHECKPOINT = args.checkpoint
MINIBATCH_SIZE = args.batch_size
VALID_STEPS_INTERVAL = args.valid_step_interval
NO_IARPA = args.no_iarpa
MODEL_MAX_LENGTH = args.model_max_length
INIT_LEARNING_RATE = args.init_learning_rate
DROPOUT = args.dropout
NUM_LAYERS = args.num_layers
WEIGHT_DECAY = args.weight_decay
TRAINING_STEPS = args.training_steps
WARMUP_STEPS = args.warmup_steps
LOSS_STR = args.loss
N_VIEWS = args.n_views
N_NEGATIVES = args.n_negatives
DO_UNFREEZE = args.do_unfreeze
UNFREEZE_LAYER_LIMIT = args.unfreeze_layer_limit
UNFREEZE_DIRECTION = args.unfreeze_direction
UNFROZEN_LEARNING_RATE = args.unfrozen_learning_rate
UNFREEZE_STEP_INTERVAL = args.unfreeze_step_interval
USE_ADAPTERS = args.transformer_adapter
CROSS_GENRE = args.cross_genre
VCROSS_GENRE = args.vcross_genre
MONITOR = args.monitor
MONITOR_MODE = args.monitor_mode
CHUNK_LENGTH = args.chunk_length
INIT_TEMPERATURE = args.init_temperature
args.num_to_multisample = list(args.num_to_multisample)
# Ensure that the list will always be 2 elements (duplicated if only one is passed to perform fixed length sampling)
NUM_TO_MULTISAMPLE = (args.num_to_multisample + args.num_to_multisample[0:1])[:2]
SAMPLED_ENCODER_LAYERS = list(args.sampled_encoder_layers)

if MODEL_TYPE == "lstm":
    MODEL = ContrastiveLSTMHead
elif MODEL_TYPE == "gru":
    MODEL = ContrastiveGRUHead
elif MODEL_TYPE == 'transformer':
    MODEL = ContrastiveTransformerHead
elif MODEL_TYPE == "projection_lstm":
    MODEL = ContrastiveLSTMProjectionHead
elif MODEL_TYPE == "long_lstm":
    MODEL = ContrastiveLSTMLongHead
elif MODEL_TYPE == "long_attention":
    MODEL = ContrastiveLSTMAttentionHead

TRAIN_FILES = {'books': 'local_data/book_train.csv',
               'mails': 'local_data/mail_train.csv',
               'blogs': 'data/nlp/blog_corpus/blog_train.csv',
               'blogs_caps': 'data/nlp/blog_corpus/blog_train_deberta.csv',
               'blogs_BigBird': 'data/nlp/blog_corpus/blog_BigBird_train.csv',
               'reddit': 'data/nlp/reddit_corpus/reddit_train.csv',
               'cross_genre': 'data/nlp/subreddit_hrs/reddit_crossgenre_deberta_train.csv',
               'hrs' : 'data/nlp/hrs_corpus/hrs_train.csv'
               }
TEST_FILES = {'books': 'local_data/book_test.csv',
              'mails': 'local_data/mail_test.csv',
              'blogs': 'data/nlp/blog_corpus/blog_test.csv',
              'blogs_caps': 'data/nlp/blog_corpus/blog_test_deberta.csv',
              'blogs_BigBird': 'data/nlp/blog_corpus/blog_BigBird_test.csv',
              'reddit': 'data/nlp/reddit_corpus/reddit_test.csv',
              'cross_genre': 'data/nlp/subreddit_hrs/reddit_crossgenre_deberta_test.csv',
              'hrs' : 'data/nlp/hrs_corpus/hrs_test.csv'
              }
USED_FILES = []

if args.books:
    USED_FILES.append('books')
if args.blogs_caps:
    USED_FILES.append('blogs_caps')
if args.blogs_BigBird:
    USED_FILES.append('blogs_BigBird')
if args.cross_genre:
    USED_FILES.append('cross_genre')
if args.mails:
    USED_FILES.append('mails')
if args.blogs:
    USED_FILES.append('blogs')
if args.reddit:
    USED_FILES.append('reddit')
if args.hrs:
    USED_FILES.append('hrs')

print(USED_FILES)
###############################################################################
# Main method #################################################################
###############################################################################


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_CODE, max_length=MODEL_MAX_LENGTH)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    # Load preferred datasets
    train_datasets, test_datasets = [], []
    tqdm.pandas()
    for file_code in USED_FILES:
        print(f'Loading {file_code} dataset...')
        train_file = pd.read_csv(TRAIN_FILES[file_code])
        test_file = pd.read_csv(TEST_FILES[file_code])

        train_file['unique_id'] = train_file.index.astype(
            str) + f'_{file_code}'
        test_file['unique_id'] = test_file.index.astype(str) + f'_{file_code}'

        dataset_columns = ['unique_id', 'id', 'decoded_text']
        if CROSS_GENRE:
            dataset_columns.append('genre')

        train_datasets.append(
            train_file[dataset_columns])
        test_datasets.append(
            test_file[dataset_columns])

    train = pd.concat(train_datasets).sample(frac=1.)
    test = pd.concat(test_datasets)

    del train_datasets
    del test_datasets

    assert (not (CROSS_GENRE and LOSS_STR == 'supcon'))
    if CROSS_GENRE:
        print('Datasets must inlcude genre columns')
        train_data = build_genre_sampling_dataset(train,
                                                  steps=TRAINING_STEPS,
                                                  tokenizer=tokenizer,
                                                  max_len=MODEL_MAX_LENGTH,
                                                  batch_size=BATCH_SIZE,
                                                  num_workers=8,
                                                  prefetch_factor=4,
                                                  shuffle=False)
    elif LOSS_STR == 'supcon':
        train_data = build_supervised_dataset(train,
                                              steps=TRAINING_STEPS,
                                              tokenizer=tokenizer,
                                              max_len=MODEL_MAX_LENGTH,
                                              batch_size=BATCH_SIZE,
                                              views=N_VIEWS,
                                              num_workers=8,
                                              prefetch_factor=4,
                                              shuffle=False)
    else:
        train_data = build_dataset(train,
                                   steps=TRAINING_STEPS,
                                   batch_size=BATCH_SIZE,
                                   num_workers=8,
                                   prefetch_factor=8,
                                   max_len=MODEL_MAX_LENGTH,
                                   tokenizer=tokenizer,
                                   shuffle=True,
                                   num_to_multisample=NUM_TO_MULTISAMPLE
                                   )
    if VCROSS_GENRE:
        test_data = build_genre_sampling_dataset(test,
                                                 steps=50,
                                                 batch_size=VALID_BATCH_SIZE,
                                                 num_workers=8,
                                                 prefetch_factor=8,
                                                 max_len=MODEL_MAX_LENGTH,
                                                 tokenizer=tokenizer,
                                                 shuffle=False)
    else:
        test_data = build_dataset(test,
                                #   steps=50,
                                  batch_size=VALID_BATCH_SIZE,
                                  num_workers=8,
                                  prefetch_factor=8,
                                  max_len=MODEL_MAX_LENGTH,
                                  tokenizer=tokenizer,
                                  shuffle=False,
                                  num_to_multisample=NUM_TO_MULTISAMPLE
                                  )

    # Name model
    model_datasets = '+'.join(USED_FILES)
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'final_{date_time}'

    print(f'Saving model to {save_name}')
    # Callbacks
    # if not args.wandb_disable:
    wandb.init(#entity="author-edu",
                project='part'  # CrossGenre
                #id=save_name,
                )
    wandb.login()
    logger = WandbLogger(name=save_name, project="part")
    # else:
    #     logger = True

    checkpoint_callback = ModelCheckpoint(
        dirpath='model',
        filename=save_name,
        monitor=MONITOR,
        mode=MONITOR_MODE,
        save_on_train_epoch_end=False,
        save_top_k=1,
    )
    
    lr_monitor = LearningRateMonitor('step')
    # Define training arguments
    trainer = Trainer(devices=DEVICES,
                      max_steps=TRAINING_STEPS,
                      accelerator='cuda',
                      log_every_n_steps=2,
                      logger=logger,
                      strategy="auto",
                      # strategy='dp',
                      precision='16-mixed',
                      val_check_interval=VALID_STEPS_INTERVAL,
                      check_val_every_n_epoch=None,
                      limit_val_batches=50,
                      num_sanity_val_steps=0,
                      #   accumulate_grad_batches=8,
                      callbacks=[checkpoint_callback, lr_monitor]
                      )

    config = {
        "init_learning_rate": INIT_LEARNING_RATE,
        "unfrozen_learning_rate": UNFROZEN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "num_warmup_steps": WARMUP_STEPS,
        "num_training_steps": TRAINING_STEPS,
        "unfreeze_step_interval": UNFREEZE_STEP_INTERVAL,
        "enable_scheduler": ENABLE_SCHEDULER,
        "minibatch_size": MINIBATCH_SIZE,
        "loss_str": LOSS_STR,
        "tokenizer": tokenizer,
        "num_head_layers": NUM_LAYERS,
        "do_unfreeze": DO_UNFREEZE,
        "unfreeze_layer_limit": UNFREEZE_LAYER_LIMIT,
        "unfreeze_direction": UNFREEZE_DIRECTION,
        "adapter_model": USE_ADAPTERS,
        "max_length": MODEL_MAX_LENGTH,
        "head_hidden_size": HEAD_HIDDEN_SIZE,
        "head_input_size": HEAD_INPUT_SIZE,
        "no_iarpa": NO_IARPA,
        "final_output_dimension": PROJECTION_OUTPUT_SIZE,
        "chunk_length": CHUNK_LENGTH,
        "dropout": DROPOUT,
        "sampled_encoder_layers": SAMPLED_ENCODER_LAYERS,
        "init_temperature": INIT_TEMPERATURE,

    }

    if CHECKPOINT:
        train_model = MODEL.load_from_checkpoint(CHECKPOINT)
        train_model.eval()
        # Update the attributes
        for key, value in config.items():
            setattr(train_model, key, value)

        # Optionally reset optimizers
        train_model.configure_optimizers()

    else:
        # Define model
        if ('T0' in BASE_CODE) or ('t5' in BASE_CODE):
            base_transformer = T5EncoderModel.from_pretrained(
                BASE_CODE, load_in_8bit=True, device_map='auto')
        elif 'gptq' in BASE_CODE.lower():
            base_transformer = AutoModelForCausalLM.from_pretrained(BASE_CODE,
                                                         use_flash_attention_2=True,
                                                         device_map='auto',
                                                         trust_remote_code=True,
                                                         output_hidden_states=True,
                                                         )
        else:
            base_transformer = AutoModel.from_pretrained(BASE_CODE, output_hidden_states=True)#,hidden_dropout_prob=DROPOUT,attention_probs_dropout_prob=DROPOUT)
        if USE_ADAPTERS:
            lora_cfg = LoraConfig(
                r=8,
                inference_mode=False,
                lora_alpha=32,
                lora_dropout=0.05,
                bias='none',
                target_modules=["query", "value"],
                # modules_to_save= ["embed_tokens", "lm_head"],
                task_type="FEATURE_EXTRACTION",
                fan_in_fan_out=False,
            )

            # base_transformer = prepare_model_for_kbit_training(base_transformer)
            base_transformer = get_peft_model(base_transformer, lora_cfg)
            base_transformer.config.gradient_checkpointing = True

        train_model = MODEL(base_transformer, **config)

    if not args.wandb_disable:
        logger.watch(train_model)

    # Fit and log
    # trainer.validate(train_model, test_data)
    trainer.fit(train_model, train_data, test_data)

    assert (trainer.checkpoint_callback.best_model_score is not None)
    print(trainer.checkpoint_callback.best_model_score.item())
    wandb.log({f"{MONITOR}": trainer.checkpoint_callback.best_model_score.item()})

    wandb.finish()


if __name__ == '__main__':
    main()
