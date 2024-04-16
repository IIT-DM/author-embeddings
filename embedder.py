from typing import Any
import torch

from model import ContrastiveLSTMHead as ModelHead

class Embedder(object):
    """standalone class that takes in one document in hts format and returns its embedding vector"""

    def __init__(self, model_path):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModelHead.load_from_checkpoint(model_path).cuda(self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer


    def __call__(self, text):
        
        tokenized = self.tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        embedding = self.model(tokenized['input_ids'], tokenized["attention_mask"]).detach().cpu().numpy().squeeze()

        return embedding

if __name__ == "__main__":

    # create callable Decoder class
    E = Embedder(model_path="/share/lvegna/Repos/author/authorship-embeddings/model/final_2023-09-05_16-06-45_lstm_blogs_electra-large-discriminator_infoNCE.ckpt")
    
    # define/get data
    hts_document = {
        "fullText": "Call me Ishmael. Some years ago—never mind how long precisely—having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people’s hats off—then, I account it high time to get to sea as soon as I can. This is my substitute for pistol and ball. With a philosophical flourish Cato throws himself upon his sword; I quietly take to the ship. There is nothing surprising in this. If they but knew it, almost all men in their degree, some time or other, cherish very nearly the same feelings towards the ocean with me.",
        "documentID": "Moby_Dick"
    }

    # call on Embedder
    embedding = E(hts_document)
    print(embedding)
    print(embedding.shape)
