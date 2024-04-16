import pandas as pd
import numpy as np


# Utilities
from sklearn.metrics import top_k_accuracy_score, accuracy_score
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from random import shuffle

TOKENIZER = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')

def embed(model, texts):
    tokenized_texts = TOKENIZER(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    embedding = model(tokenized_texts.input_ids.to(model.device),
                      tokenized_texts.attention_mask.to(model.device),
                      )
    return embedding

def embed_transformer(model, texts):
    tokenized_texts = TOKENIZER(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    embedding = model(tokenized_texts.input_ids.to(model.device),
                      attention_mask = tokenized_texts.attention_mask.to(model.device),
                      ).pooler_output
    return embedding


def evaluate(model, data, top_k=5, N=100, repetitions=1, embed_f=embed):
    with torch.no_grad():
        
        accs, topks = [], []
        for _ in tqdm(range(repetitions)):           
            authors = data.id.unique().tolist()
            shuffle(authors)
            random_authors = authors[:N]
            anchors, replicas = [], []
            for author in random_authors:
                anchor, replica = data.loc[author == data.id].decoded_text.sample(2).tolist()
                anchors.append(anchor)
                replicas.append(replica)
            
            embedding_anchors = F.normalize(embed_f(model, anchors))
            embedding_replicas = F.normalize(embed_f(model, replicas))

            preds = embedding_anchors @ embedding_replicas.T
            labels = torch.arange(0, len(preds)).numpy()

            preds_a = F.softmax(preds, dim=-1)
            preds_b = F.softmax(preds.T, dim=-1)

            a_acc = accuracy_score(labels, preds_a.argmax(-1).cpu().numpy())
            b_acc = accuracy_score(labels, preds_b.argmax(-1).cpu().numpy())
            a_topk = top_k_accuracy_score(y_true=labels, y_score=preds_a.cpu().numpy(), k=top_k)
            b_topk = top_k_accuracy_score(y_true=labels, y_score=preds_b.cpu().numpy(), k=top_k)

            accs.append((a_acc+b_acc)/2)
            topks.append((a_topk+b_topk)/2)

            del embedding_anchors
            del embedding_replicas

            torch.cuda.empty_cache()
            
        return np.mean(accs), np.mean(topks), np.std(accs), np.std(topks)

def evaluate_crossgenre(model, data, top_k=5, N=100, repetitions=1, embed_f=embed):
    with torch.no_grad():
        accs, topks = [], []
        for _ in tqdm(range(repetitions)):
            authors = data.id.unique().tolist()
            anchors, replicas = [], []

            shuffle(authors)
            for author in authors:
                if len(anchors) >= N:
                    break

                # Filter data for the current author
                author_data = data.loc[data.id == author]
                # Ensure the author has more than one genre
                if author_data.genre.nunique() > 1:
                    # Randomly select an anchor
                    anchor = author_data.sample(1)
                    anchor_text = anchor.decoded_text.iloc[0]
                    anchor_genre = anchor.genre.iloc[0]

                    # Filter out examples of the same genre
                    different_genre_data = author_data.loc[author_data.genre != anchor_genre]
                    if not different_genre_data.empty:
                        # Randomly select a replica from the different genre
                        replica = different_genre_data.sample(1).decoded_text.iloc[0]

                        anchors.append(anchor_text)
                        replicas.append(replica)
                else:
                    print(author_data)
            if len(anchors) < N:
                print(f"WARNING: Only found {len(anchors)} pairs, this may cause biased results!")

            embedding_anchors = F.normalize(embed_f(model, anchors))
            embedding_replicas = F.normalize(embed_f(model, replicas))

            preds = embedding_anchors @ embedding_replicas.T
            labels = torch.arange(0, len(preds)).numpy()

            preds_a = F.softmax(preds, dim=-1)
            preds_b = F.softmax(preds.T, dim=-1)

            a_acc = accuracy_score(labels, preds_a.argmax(-1).cpu().numpy())
            b_acc = accuracy_score(labels, preds_b.argmax(-1).cpu().numpy())
            a_topk = top_k_accuracy_score(y_true=labels, y_score=preds_a.cpu().numpy(), k=top_k)
            b_topk = top_k_accuracy_score(y_true=labels, y_score=preds_b.cpu().numpy(), k=top_k)

            accs.append((a_acc + b_acc) / 2)
            topks.append((a_topk + b_topk) / 2)

            del embedding_anchors
            del embedding_replicas

            torch.cuda.empty_cache()

        return np.mean(accs), np.mean(topks), np.std(accs), np.std(topks)



test_blogs = pd.read_csv('/share/lvegna/Repos/author/authorship-embeddings/data/nlp/subreddit_hrs/reddit_crossgenre_deberta_test.csv')
model_zoo = {#'all': 'model/final_2022-06-21_12-22-26_lstm_books+mails+blogs.ckpt',
            # 'books': 'model/final_2022-06-28_07-55-16_lstm_books.ckpt',
            # 'mails': 'model/final_2022-06-27_08-14-08_lstm_mails.ckpt',
            'blogs': '/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-03-14_11-31-19.ckpt',
            'crossgenre': '/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-03-14_11-31-19.ckpt'

            #'/share/lvegna/Repos/author/authorship-embeddings/model/final_2024-01-22_16-46-10.ckpt',
            }
data_zoo = {#'all': test_books.append(test_mails).append(test_blogs),
            #'books': test_books,
            #'mails': test_mails,
            'blogs': test_blogs,
            }

from model import (ContrastiveMaxDenseHead,
                   ContrastiveMeanDenseHead, 
                   ContrastiveLSTMHead,
                   )
from model import (ContrastiveMeanTransformer,
                                # ContrastiveLSTMTransformer,
                                # ContrastiveMeanDenseTransformer,
                                # ContrastiveMaxDenseTransformer,
                                 )
from random import shuffle

REPEATS = 50
TOP_K = 5
DEVICE = 0

n_list = [8, 16, 32, 64]
# keys = ['all', 'blogs', 'books', 'mails']
keys = ['blogs']

data_dict = {}
# Pass on trained models
for k_model in keys:
    model_ckpt = model_zoo[k_model]
    model = ContrastiveLSTMHead.load_from_checkpoint(model_ckpt).cuda(DEVICE)
    # print(model.state_dict().keys())
    # print(model.hparams)

    # exit()
    data_dict[f'Model_{k_model}'] = {}
    print(f'[log] Evaluating model - {k_model}')

    for k_data in keys:
        data = data_zoo[k_data]
        
        data_dict[f'Model_{k_model}'][f'data_{k_data}'] = {}
        print(f'[log] Evaluating on data - {k_data}')

        for n in n_list:
            print(f'[log] Running {REPEATS} repetitions for N={n}')
            max_len = len(data.id.unique())
            
            if n == 'max':
                n = max_len

            if n <= max_len:
                acc, topk, acc_sd, topk_sd = evaluate_crossgenre(model, data, top_k=TOP_K, N=n, repetitions=REPEATS)
                data_dict[f'Model_{k_model}'][f'data_{k_data}'][f'N={n} Accuracy'] = f'{100*acc:.2f}% ± {100*acc_sd:.2f}'
                data_dict[f'Model_{k_model}'][f'data_{k_data}'][f'N={n} Top-5 Accuracy'] = f'{100*topk:.2f}% ± {100*topk_sd:.2f}'

                print(f'[log] model:{k_model}_data:{k_data}_n:{n}: {100*acc:.2f}% {100*topk:.2f}%')
    
    del model
    torch.cuda.empty_cache()


import json
with open('/share/lvegna/Repos/author/authorship-embeddings/results/temp.json', 'w') as f:
    json.dump(data_dict, f)
