import torch

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

import pytorch_lightning as pl
from modules import DynamicLSTM, DynamicGRU, SimpleTransformer
from iarpa5.eval import run_iarpa5_eval
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
import torch.nn as nn
import torch.nn.functional as F
import math

class ContrastivePretrain(pl.LightningModule):

    def switch_finetune(self, switch=True):
        for name, param in self.transformer.named_parameters():
                param.requires_grad = switch
          
                
                
    def disable_non_adapters(self):
        from peft.tuners.tuners_utils import BaseTunerLayer
        for _, module in self.transformer.named_modules():
            if not isinstance(module, BaseTunerLayer):
                print(f"disabling:\n {module}")
                module.requires_grad = False
                for param in module:
                    param.requires_grad = False
                
    
    @staticmethod
    def unfreeze_layers(layers, switch=True):
        for param in layers.parameters():
            param.requires_grad = switch

    def custom_scheduler(self, step):
        # If do_unfreeze is False, just do a warmup to the init_learning_rate
        if not self.do_unfreeze:
            if step < self.num_warmup_steps:
                warmup_ratio = step / self.num_warmup_steps
                return 0.01 * self.init_learning_rate + warmup_ratio * 0.99 * self.init_learning_rate
            else:
                return self.init_learning_rate

        # Calculate how many unfreeze intervals have passed
        unfreeze_intervals_passed = step // self.unfreeze_step_interval

        # Determine if we're still in the warmup phase for the current interval
        if step % self.unfreeze_step_interval < self.num_warmup_steps:
            warmup_ratio = (step % self.unfreeze_step_interval) / self.num_warmup_steps
        else:
            warmup_ratio = 1.0

        # Set learning rate based on the number of unfreeze intervals passed and the warmup ratio
        if unfreeze_intervals_passed <= self.unfreeze_layer_limit:
            if unfreeze_intervals_passed == 0:
                return 0.01 * self.init_learning_rate + warmup_ratio * 0.99 * self.init_learning_rate
            else:
                # Compute the learning rate based on warmup for unfrozen_learning_rate
                return 0.01 * self.unfrozen_learning_rate + warmup_ratio * 0.99 * self.unfrozen_learning_rate
        else:
            # If we've reached or exceeded the unfreeze layer limit, maintain the unfrozen learning rate
            return self.unfrozen_learning_rate


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1,
                          weight_decay=self.weight_decay)

        scheduler = LambdaLR(
            optimizer, lr_lambda=lambda step: self.custom_scheduler(step))
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": 'step',  # Make sure it's updated on a per-step basis
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": 'custom_scheduler',
        }

        if self.enable_scheduler:
            return {'optimizer': optimizer,
                    'lr_scheduler': lr_scheduler_config,
                    }
        else:
            return {'optimizer': optimizer}

    def infonce_loss(self, a, b):
        batch_size = a.shape[0]
        logits = (a @ b.T) * torch.exp(self.temperature).clamp(max=100)
        labels = torch.arange(0, batch_size, device=self.device)

        loss = (F.cross_entropy(logits.T, labels).mean() +
                F.cross_entropy(logits, labels).mean()) / 2

        with torch.no_grad():
            preds = F.softmax(logits, dim=1).argmax(-1)
            preds_t = F.softmax(logits.T, dim=1).argmax(-1)

            accuracy = (torch.sum(preds == labels) +
                        torch.sum(preds_t == labels)) / (batch_size * 2)

        return loss, accuracy

    def supcon_loss(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """

        batch_size = features.shape[0]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(features.device)

        views = features.shape[1]  # = n_views
        full_features = torch.cat(torch.unbind(
            features, dim=1), dim=0)  # = [bsz*views, ...]
        
        
        # compute logits (cosine sim)
        anchor_dot_contrast = torch.matmul(F.normalize(full_features),
                                           F.normalize(full_features.T)) * torch.exp(self.temperature).clamp(100)  # = [bsz*views, bsz*views]

        loss_0 = self._loss_from_dot(
            anchor_dot_contrast, mask, views, batch_size)
        loss_1 = self._loss_from_dot(
            anchor_dot_contrast.T, mask.T, views, batch_size)

        return (loss_0 + loss_1) / 2


    def _loss_from_dot(self, anchor_dot_contrast, mask, views, batch_size):  # (anchor, contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(views, views)
        # mask-out self-contrast cases
        logits_mask = 1 - torch.eye(views*batch_size, device=mask.device)
        mask = mask * logits_mask
        
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - \
            torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.view(views, batch_size).mean()

        return loss

    def infonce_loss_euclidean(self, a, b):
        batch_size = a.shape[0]
        print(a)
        print(b)
        # Use negative euclidean distance as logits (smaller distances will lead to larger logits)
        logits = -self._euclidean_dist(a, b) * \
            torch.exp(self.temperature).clamp(max=100)
        labels = torch.arange(0, batch_size, device=self.device)

        loss = (F.cross_entropy(logits.T, labels).mean() +
                F.cross_entropy(logits, labels).mean()) / 2

        with torch.no_grad():
            preds = F.softmax(logits, dim=1).argmax(-1)
            preds_t = F.softmax(logits.T, dim=1).argmax(-1)

            accuracy = (torch.sum(preds == labels) +
                        torch.sum(preds_t == labels)) / (batch_size * 2)

        return loss, accuracy

    


    def hard_negatives_supcon_loss(self, features):

        anchor = features[:, [0], ...]  # [bsz, 1, dim]
        replica_and_negatives = features[:, 1:, ...]  # [bsz, 1+n, dim]

        logits = (replica_and_negatives @ anchor.swapdims(1, 2)
                  ).squeeze()  # [bsz, 1+n]
        logits = logits * torch.exp(self.temperature).clamp(max=100)

        labels = torch.zeros(
            (anchor.shape[0],), dtype=torch.long, device=self.device)  # [1+n]
        print(logits)
        print(labels)
        loss = F.cross_entropy(logits, labels).mean()
        print(loss)
        exit()
        return loss

    def _euclidean_dist(self, x, y):
        # Compute pairwise Euclidean distances between x and y
        m, n = x.size(0), y.size(0)
        xx = (x**2).sum(dim=1, keepdim=True).expand(m, n)
        yy = (y**2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def eval_batch(self, batch, loss_str, mode='train'):

        anchors_input_ids, anchors_attention_mask, replicas_input_ids, replicas_attention_mask = batch
        anchor_embeds = self(anchors_input_ids, anchors_attention_mask)
        replicas_embeds = self(replicas_input_ids, replicas_attention_mask)

        if loss_str == "infoNCE_euclidean":
            loss, acc = self.infonce_loss_euclidean(
                anchor_embeds, replicas_embeds)
        else:
            loss, acc = self.infonce_loss(anchor_embeds, replicas_embeds)

        self.log(f'{mode}/infonce_loss', loss)
        self.log(f'{mode}/infonce_acc', acc)

        return loss


    def eval_batch_supervised(self, batch, mode='train'):
        ids, masks, labels = batch
        bsz, n_views = ids.shape[0], ids.shape[1]

        embeds = self(ids.view(bsz * n_views, -1),
                    masks.view(bsz * n_views, -1)).reshape((bsz, n_views, -1))

        loss = self.supcon_loss(embeds, labels)
        
        # Compute the mean positive pairwise distance metric
        
        # Log the values
        self.log(f'{mode}/supcon_loss', loss)

        return loss


    def eval_batch_hard_negatives(self, batch, mode='train'):

        ids, masks, _ = batch
        bsz, n = ids.shape[0], ids.shape[1]

        embeds = self(ids.view(bsz * n, -1),
                      masks.view(bsz * n, -1)).reshape((bsz, n, -1))
        loss = self.hard_negatives_supcon_loss(embeds)

        self.log(f'{mode}/hard_negatives_loss', loss)

        return loss

    

    # Progressively unfreeze layers in the encoder transformer at intervals of unfreeze step
    def progressive_unfreeze(self, unfreeze_layer_limit, unfreeze_direction):
        # Try to access common attributes for encoder layers.
        layers = None
        try:
            layers = self.transformer.encoder.layer
        except AttributeError:
            # For models that don't have a typical BERT-like structure.
            for name, module in self.transformer.named_children():
                if 'layer' in name:
                    layers = module
                    break

        if layers is None:
            raise ValueError("Could not find the layers in the model.")

        num_layers = len(layers)
        unfreeze_stage = self.global_step // self.unfreeze_step_interval
        num_layers_to_unfreeze = min(unfreeze_stage, unfreeze_layer_limit)

        # Depending on the unfreeze direction, determine the start and end indices
        if unfreeze_direction > 0:
            start_layer_idx = 0
            end_layer_idx = num_layers_to_unfreeze
        else:
            start_layer_idx = num_layers - num_layers_to_unfreeze
            end_layer_idx = num_layers

        # Only unfreeze when it's time (based on global_step and unfreeze interval)
        if self.global_step != 0 and self.global_step % self.unfreeze_step_interval == 0:
            for idx in range(start_layer_idx, end_layer_idx):
                if 0 <= idx < num_layers:  # Ensure the index is within the valid range.
                    print(f"Unfreezing layer {idx}")
                    self.unfreeze_layers(layers[idx], True)

    def training_step(self, train_batch, batch_idx):
        if self.do_unfreeze:
            self.progressive_unfreeze(self.unfreeze_layer_limit, self.unfreeze_direction)
        if self.loss_str == "supcon":
            return self.eval_batch_supervised(train_batch)
        elif self.loss_str == "hard_negatives":
            return self.eval_batch_hard_negatives(train_batch)
        return self.eval_batch(train_batch, loss_str=self.loss_str)

    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            if batch_idx == 0:
                self.log("temperature", self.temperature.detach())
                if not(self.no_iarpa):
                    iarpa_results = run_iarpa5_eval(
                        self, device=self.device, distance='euclidean')
                    self.log_dict(iarpa_results)
            return self.eval_batch(val_batch, loss_str=self.loss_str, mode='valid')

    def test_step(self, test_batch, batch_idx):
        return self.eval_batch(test_batch, loss_str=self.loss_str, mode='test')

    def predict_step(self, pred_batch, batch_idx):
        anchors, _, _ = pred_batch

        return self(anchors.input_ids, anchors.attention_mask)


class ContrastiveLSTMHead(ContrastivePretrain):
    def __init__(self, transformer=None,
                 init_learning_rate=5e-3,
                 weight_decay=.01,
                 num_warmup_steps=0,
                 num_training_steps=8000,
                 enable_scheduler=False,
                 head_hidden_size=128,
                 head_input_size=256,
                 num_head_layers=1,
                 loss_str='infoNCE',
                 tokenizer=None,
                 do_unfreeze=False,
                 adapter_model=False,
                 max_length = 512,
                 **kwargs,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.init_learning_rate = init_learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler
        self.head_hidden_size = head_hidden_size
        self.head_input_size = head_input_size
        self.loss_str = loss_str
        self.num_head_layers = num_head_layers
        self.do_unfreeze = do_unfreeze
        self.adapter_model = adapter_model
        self.init_temperature = kwargs.get("init_temperature", 3.0)
        self.unfreeze_layer_limit = kwargs.get('unfreeze_layer_limit', None)
        self.unfrozen_learning_rate = kwargs.get('unfrozen_learning_rate', None)
        self.unfreeze_direction = kwargs.get('unfreeze_direction', None)
        self.unfreeze_step_interval = kwargs.get('unfreeze_step_interval', None)
        self.no_iarpa = kwargs.get('no_iarpa', None)
        self.dropout = kwargs.get('dropout', 0.1)
        self.sampled_encoder_layers = kwargs.get('sampled_encoder_layers', [-1])
        self.max_length = max_length

        self.save_hyperparameters(ignore=['transformer'])

        if transformer:
            self.set_transformer(transformer, tokenizer)
            
        self.temperature = torch.nn.Parameter(torch.Tensor([self.init_temperature]))
        # self.temperature.requires_grad = False
        self.epsilon = 1e-8

    def set_transformer(self, transformer, tokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.max_length
        
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(transformer.config.hidden_size,
                            self.head_input_size),
            torch.nn.ReLU()
        )
        self.pooler = DynamicLSTM(self.head_input_size,
                                  self.head_hidden_size,
                                  num_layers=self.num_head_layers,
                                  dropout=self.dropout,
                                  bidirectional=True)
        
        self.switch_finetune(False)  

    def on_save_checkpoint(self, checkpoint):
        checkpoint['transformer_state_dict'] = self.transformer.state_dict()
        checkpoint['transformer_config'] = self.transformer.config
        checkpoint['tokenizer_state'] = self.tokenizer

    def on_load_checkpoint(self, checkpoint):
        # Instantiate the transformer from the saved config
        self.transformer = AutoModel.from_config(
            checkpoint['transformer_config'])
        self.tokenizer = checkpoint['tokenizer_state']

        # Load state dict
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.set_transformer(self.transformer, self.tokenizer)

    def forward(self, input_ids, attention_mask=None):
        output = self.transformer(input_ids, attention_mask)
        
        # Check if hidden_states are available
        if 'hidden_states' in output:
            # Extract the layers specified in sampled_encoder_layers
            sampled_layers = [output.hidden_states[layer] for layer in self.sampled_encoder_layers]
            # Stack the sampled layers and compute their mean
            embeds = torch.mean(torch.stack(sampled_layers), dim=0)
        else:
            # Fallback to the last hidden state if hidden_states are not available
            embeds = output.last_hidden_state

        # Apply the projection
        embeds = self.projection(embeds)

        # Pool the projected embeddings
        pooled_embed = self.pooler(embeds, attention_mask)

        # Return based on the loss strategy
        if self.loss_str == 'infoNCE_euclidean':
            return pooled_embed
        else:
            return F.normalize(pooled_embed)

class ContrastiveGRUHead(ContrastivePretrain):
    def __init__(self, transformer=None,
                 init_learning_rate=5e-3,
                 weight_decay=.01,
                 num_warmup_steps=0,
                 num_training_steps=8000,
                 enable_scheduler=False,
                 head_hidden_size=128,
                 head_input_size=256,
                 num_head_layers=1,
                 loss_str='infoNCE',
                 tokenizer=None,
                 do_unfreeze=False,
                 adapter_model=False,
                 max_length = 512,
                 **kwargs,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.init_learning_rate = init_learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler
        self.head_hidden_size = head_hidden_size
        self.head_input_size = head_input_size
        self.loss_str = loss_str
        self.num_head_layers = num_head_layers
        self.do_unfreeze = do_unfreeze
        self.adapter_model = adapter_model
        self.init_temperature = kwargs.get("init_temperature", 3.0)
        self.unfreeze_layer_limit = kwargs.get('unfreeze_layer_limit', None)
        self.unfrozen_learning_rate = kwargs.get('unfrozen_learning_rate', None)
        self.unfreeze_direction = kwargs.get('unfreeze_direction', None)
        self.unfreeze_step_interval = kwargs.get('unfreeze_step_interval', None)
        self.no_iarpa = kwargs.get('no_iarpa', None)
        self.dropout = kwargs.get('dropout', 0.1)
        self.sampled_encoder_layers = kwargs.get('sampled_encoder_layers', [-1])
        self.max_length = max_length

        self.save_hyperparameters(ignore=['transformer'])

        if transformer:
            self.set_transformer(transformer, tokenizer)
            
        self.temperature = torch.nn.Parameter(torch.Tensor([self.init_temperature]))
        # self.temperature.requires_grad = False
        self.epsilon = 1e-8

    def set_transformer(self, transformer, tokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.max_length
        
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(transformer.config.hidden_size,
                            self.head_input_size),
            torch.nn.ReLU()
        )
        self.pooler = DynamicGRU(self.head_input_size,
                                  self.head_hidden_size,
                                  num_layers=self.num_head_layers,
                                  dropout=self.dropout,
                                  bidirectional=True)
        
        self.switch_finetune(False)  

    def on_save_checkpoint(self, checkpoint):
        checkpoint['transformer_state_dict'] = self.transformer.state_dict()
        checkpoint['transformer_config'] = self.transformer.config
        checkpoint['tokenizer_state'] = self.tokenizer

    def on_load_checkpoint(self, checkpoint):
        # Instantiate the transformer from the saved config
        self.transformer = AutoModel.from_config(
            checkpoint['transformer_config'])
        self.tokenizer = checkpoint['tokenizer_state']

        # Load state dict
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.set_transformer(self.transformer, self.tokenizer)

    def forward(self, input_ids, attention_mask=None):
        output = self.transformer(input_ids, attention_mask)
        
        # Check if hidden_states are available
        if 'hidden_states' in output:
            # Extract the layers specified in sampled_encoder_layers
            sampled_layers = [output.hidden_states[layer] for layer in self.sampled_encoder_layers]
            # Stack the sampled layers and compute their mean
            embeds = torch.mean(torch.stack(sampled_layers), dim=0)
        else:
            # Fallback to the last hidden state if hidden_states are not available
            embeds = output.last_hidden_state

        # Apply the projection
        embeds = self.projection(embeds)

        # Pool the projected embeddings
        pooled_embed = self.pooler(embeds, attention_mask)

        # Return based on the loss strategy
        if self.loss_str == 'infoNCE_euclidean':
            return pooled_embed
        else:
            return F.normalize(pooled_embed)



class ContrastiveLSTMProjectionHead(ContrastiveLSTMHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass any additional args and kwargs to the base class
        self.final_output_dimension = kwargs.get("final_output_dimension", 256)
        
        self.output_projection = torch.nn.Linear(self.head_hidden_size * 2, self.final_output_dimension)
        
        
    def forward(self, input_ids, attention_mask=None):
        embeds = self.transformer(input_ids, attention_mask).last_hidden_state
        embeds = self.projection(embeds)
        pooled_embed = self.pooler(embeds, attention_mask)
        pooled_embed = self.output_projection(pooled_embed)
        if self.loss_str == 'infoNCE_euclidean':
            return pooled_embed
        return F.normalize(pooled_embed)


class ContrastiveTransformerHead(ContrastivePretrain):
    def __init__(self, transformer=None,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=False,
                 head_hidden_size=256,
                 head_input_size=256,
                 head_depth=1,
                 loss_str='infoNCE',
                 tokenizer=None,
                 **kwargs,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler
        self.head_hidden_size = head_hidden_size
        self.head_input_size = head_input_size
        self.head_depth = head_depth
        self.loss_str = loss_str
        self.save_hyperparameters(ignore=['transformer'])

        if transformer:
            self.set_transformer(transformer, tokenizer)

        self.temperature = torch.nn.Parameter(torch.Tensor([0.07]))

    def set_transformer(self, transformer, tokenizer):
        self.transformer = transformer
        self.tokenizer = tokenizer
        # embed_size = transformer.config.hidden_size // 2
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(transformer.config.hidden_size,
                            self.head_input_size),
            torch.nn.ReLU()
        )
        self.pooler = SimpleTransformer(
            d_model=self.head_hidden_size, num_layers=self.head_depth, dropout=.05, pooling='mean')
        self.switch_finetune(False)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['transformer_state_dict'] = self.transformer.state_dict()
        checkpoint['transformer_config'] = self.transformer.config
        checkpoint['tokenizer_state'] = self.tokenizer

    def on_load_checkpoint(self, checkpoint):
        # Instantiate the transformer from the saved config
        self.transformer = AutoModel.from_config(
            checkpoint['transformer_config'])
        self.tokenizer = checkpoint['tokenizer_state']
        # Load state dict
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.set_transformer(self.transformer, self.tokenizer)

    # def forward(self, input_ids, attention_mask=None):
    #     embeds = self.transformer(input_ids, attention_mask).last_hidden_state
    #     embeds = self.projection(embeds)
    #     return F.normalize(self.pooler(embeds, attention_mask))

    def forward(self, input_ids, attention_mask=None):
        # Checkpointing the transformer part. This will only be relevant if weights are unfrozen
        def run_transformer(input_ids, attention_mask):
            return self.transformer(input_ids, attention_mask).last_hidden_state

        embeds = checkpoint(run_transformer, embeds)

        # Checkpointing the projection part
        def run_projection(embeds):
            return self.projection(embeds)

        embeds_checkpointed = checkpoint(run_projection, embeds)

        # Checkpointing the pooler (DynamicLSTM)
        def run_pooler(embeds_checkpointed, attention_mask):
            return self.pooler(embeds_checkpointed, attention_mask)

        pooled_output = checkpoint(
            run_pooler, embeds_checkpointed, attention_mask)

        return F.normalize(pooled_output)


class ContrastiveTransformer(ContrastivePretrain):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=True,
                 **kwargs,
                 ):
        super().__init__()

        # Save hyperparameters for training

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler

        self.save_hyperparameters()

        self.transformer = transformer

        embed_size = transformer.config.hidden_size
        self.pooler = torch.nn.Linear(embed_size, embed_size)
        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))


class ContrastiveMeanTransformer(ContrastiveTransformer):
    def forward(self, input_ids, attention_mask=None):
        embeds = self.transformer(input_ids, attention_mask).last_hidden_state

        if attention_mask is None:
            embed = embeds.mean(1)
        else:
            embed = (embeds*attention_mask.unsqueeze(-1)).sum(1) / \
                attention_mask.sum(1).unsqueeze(-1)

        return F.normalize(self.pooler(embed))


class ContrastiveMaxTransformer(ContrastiveTransformer):
    def forward(self, input_ids, attention_mask=None):
        embeds = self.transformer(input_ids, attention_mask).last_hidden_state
        embed = embeds.max(1)[0]

        return F.normalize(self.pooler(embed))


class ContrastiveDenseHead(ContrastivePretrain):
    def __init__(self, transformer,
                 learning_rate=5e-5,
                 weight_decay=.01,
                 num_warmup_steps=1000,
                 num_training_steps=10000,
                 enable_scheduler=False,
                 **kwargs,
                 ):
        super().__init__()

        # Save hyperparameters for training
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.enable_scheduler = enable_scheduler

        self.save_hyperparameters(ignore=['transformer'])
        self.transformer = transformer
        for param in self.transformer.parameters():
            param.requires_grad = False

        embed_size = transformer.config.hidden_size
        self.pooler = torch.nn.Linear(embed_size, embed_size)
        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))


class ContrastiveMeanDenseHead(ContrastiveDenseHead):
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeds = self.transformer(
                input_ids, attention_mask).last_hidden_state

            if attention_mask is None:
                embed = embeds.mean(1)
            else:
                embed = (embeds*attention_mask.unsqueeze(-1)).sum(1) / \
                    attention_mask.sum(1).unsqueeze(-1)

        return F.normalize(self.pooler(embed))


class ContrastiveMaxDenseHead(ContrastiveDenseHead):
    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            embeds = self.transformer(
                input_ids, attention_mask).last_hidden_state

            embed = embeds.max(1)[0]

        return F.normalize(self.pooler(embed))
    
# ======================
# TA2 Below ============
# ======================


class ContrastiveLSTMLongHead(ContrastiveLSTMHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass any additional args and kwargs to the base class
        self.chunk_length = kwargs.get("chunk_length", 512)
            
    def forward(self, input_ids, attention_mask=None):
        input_shape = input_ids.size()
        total_length = input_shape[1]

        # Split input_ids and attention_mask into chunks and process each chunk
        chunked_embeds = self.process_chunks(input_ids, attention_mask, total_length)
        
        # Pooling
        pooled_embed = self.pooler(chunked_embeds, attention_mask)
        
        # Handle the output based on the loss function
        if self.loss_str == 'infoNCE_euclidean':
            return pooled_embed
        else:
            return F.normalize(pooled_embed)

    def process_chunks(self, input_ids, attention_mask, total_length):
        pooled_output = []
        for i in range(0, total_length, self.chunk_length):
            end_idx = min(i + self.chunk_length, total_length)  # Compute the end index of the chunk
            chunk_input_ids = input_ids[:, i:end_idx]
            chunk_attention_mask = attention_mask[:, i:end_idx] if attention_mask is not None else None

            # TODO: Dont include all 0 chunk_attention_mask in the pooled outputs
            
            chunk_embeds = self.transformer(chunk_input_ids, chunk_attention_mask).last_hidden_state
            chunk_embeds = self.projection(chunk_embeds)
            pooled_output.append(chunk_embeds)

        # Concatenate the processed chunks along the sequence length dimension
        embeds = torch.cat(pooled_output, dim=1)
        return embeds


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ContrastiveLSTMAttentionHead(ContrastiveLSTMLongHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_pooler = self.__AttentionPooling(self.head_hidden_size * 2, math.ceil(self.max_length / self.chunk_length))

    def forward(self, input_ids, attention_mask):
        input_shape = input_ids.size()
        total_length = input_shape[1]
        pooled_outputs = []
        attention_masks = []  # Prepare to collect attention masks for chunks
        for i in range(0, total_length, self.chunk_length):
            end_idx = min(i + self.chunk_length, total_length)
            chunk_input_ids = input_ids[:, i:end_idx]
            chunk_attention_mask = attention_mask[:, i:end_idx] if attention_mask is not None else None
            chunk_embeds = self.transformer(chunk_input_ids, chunk_attention_mask).last_hidden_state
            chunk_embeds = self.projection(chunk_embeds)
            valid_mask = torch.sum(chunk_attention_mask, dim=1) > 0 if chunk_attention_mask is not None else torch.ones(chunk_input_ids.size(0), dtype=torch.bool, device=chunk_input_ids.device)

            # Initialize placeholder for pooled chunk embeddings with zeros
            pooled_chunk_placeholder = torch.zeros((chunk_embeds.size(0), self.head_hidden_size * 2), device=chunk_embeds.device, dtype=chunk_embeds.dtype)

            # Modified condition to check if there are any True values in valid_mask
            if valid_mask.any():
                # Process only valid sequences through the pooler
                valid_chunk_embeds = chunk_embeds[valid_mask]
                valid_chunk_attention_mask = chunk_attention_mask[valid_mask] if chunk_attention_mask is not None else None
                pooled_chunk = self.pooler(valid_chunk_embeds, valid_chunk_attention_mask)
                # Assign the pooled embeddings to the correct positions in the placeholder
                pooled_chunk_placeholder[valid_mask] = pooled_chunk
            pooled_outputs.append(pooled_chunk_placeholder.unsqueeze(1))
            
            chunk_mask = valid_mask.long().unsqueeze(-1)
            attention_masks.append(chunk_mask)


        # Concatenate the pooled chunk embeddings along the sequence length dimension
        concat_pooled_outputs = torch.cat(pooled_outputs, dim=1)
        # Concatenate the attention masks for each chunk
        attention_mask_for_chunks = torch.cat(attention_masks, dim=1)
        # Apply attention-based pooling to the concatenated pooled outputs
        final_pooled_output = self.attention_pooler(concat_pooled_outputs, attention_mask_for_chunks)
        return F.normalize(final_pooled_output)  # Add back the sequence dimension if needed

    class __AttentionPooling(nn.Module):
        def __init__(self, embedding_dim, num_heads=8):
            super().__init__()
            self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        def forward(self, x, mask=None):
            # MultiheadAttention expects mask to be True for positions not to attend to.
            # Since your mask uses 1s for valid positions and 0s for masked, we invert the mask.
            if mask is not None:
                attn_mask = mask == 0
            else:
                attn_mask = None

            # Adjusting input's shape to [sequence_length, batch_size, embedding_dim] as expected by nn.MultiheadAttention
            x = x.permute(1, 0, 2)

            # Apply MultiheadAttention
            # Note: attn_mask here is used directly following the inversion above.
            attn_output, _ = self.multihead_attn(query=x, key=x, value=x, key_padding_mask=attn_mask)

            # Convert attn_output back to [batch_size, sequence_length, embedding_dim] format
            attn_output = attn_output.permute(1, 0, 2)

            # Now, compute the mean of attended outputs. First, we'll apply the mask to ignore masked positions.
            if mask is not None:
                # Expand mask to match attn_output shape for multiplication
                mask_expanded = mask.unsqueeze(-1).type_as(attn_output)
                sum_outputs = torch.sum(attn_output * mask_expanded, dim=1)
                valid_elements = mask_expanded.sum(dim=1)
                valid_elements = valid_elements.clamp(min=1)  # Avoid division by zero
                mean_output = sum_outputs / valid_elements
            else:
                # If no mask is provided, compute the mean across the sequence dimension
                mean_output = attn_output.mean(dim=1)

            return mean_output
                
