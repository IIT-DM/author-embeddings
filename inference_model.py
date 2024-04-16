import torch

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

import pytorch_lightning as pl
import torch.nn.functional as F
from modules import DynamicLSTM
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint


class ContrastivePretrain(pl.LightningModule):

    def switch_finetune(self, switch=True):
        for param in self.transformer.parameters():
            param.requires_grad = switch

    @staticmethod
    def unfreeze_layers(layers, switch=True):
        for param in layers.parameters():
            param.requires_grad = switch

    def custom_scheduler(self, step):
        # If do_unfreeze is False, just do a warmup to the init_learning_rate
        if not self.do_unfreeze:
            if step < self.num_warmup_steps:
                warmup_ratio = step / self.num_warmup_steps
                return 0.1 * self.init_learning_rate + warmup_ratio * 0.9 * self.init_learning_rate
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
        if unfreeze_intervals_passed < self.unfreeze_layer_limit:
            if unfreeze_intervals_passed == 0:
                return 0.1 * self.init_learning_rate + warmup_ratio * 0.9 * self.init_learning_rate
            else:
                # Compute the learning rate based on warmup for unfrozen_learning_rate
                return 0.1 * self.unfrozen_learning_rate + warmup_ratio * 0.9 * self.unfrozen_learning_rate
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

        self.log(f'{mode}/supcon_loss', loss)

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
        return self.eval_batch(train_batch, loss_str=self.loss_str)

    def validation_step(self, val_batch, batch_idx):
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
        self.unfreeze_layer_limit = kwargs.get('unfreeze_layer_limit', None)
        self.unfrozen_learning_rate = kwargs.get('unfrozen_learning_rate', None)
        self.unfreeze_direction = kwargs.get('unfreeze_direction', None)
        self.unfreeze_step_interval = kwargs.get('unfreeze_step_interval', None)

        self.save_hyperparameters(ignore=['transformer'])

        if transformer:
            self.set_transformer(transformer, tokenizer)

        self.temperature = torch.nn.Parameter(torch.Tensor([.07]))
        self.epsilon = 1e-8

    def set_transformer(self, transformer, tokenizer):
        self.transformer = transformer
        print(f"Transformer device = {self.transformer.device}")
        self.tokenizer = tokenizer
        # self.tokenizer.model_max_length = 512
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(transformer.config.hidden_size,
                            self.head_input_size),
            torch.nn.ReLU()
        )
        self.pooler = DynamicLSTM(self.head_input_size,
                                  self.head_hidden_size,
                                  num_layers=self.num_head_layers,
                                  dropout=.1,
                                  bidirectional=True)
        print(f"Pooler device = {self.pooler.device}")
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
        embeds = self.transformer(input_ids, attention_mask).last_hidden_state
        embeds = self.projection(embeds)
        pooled_embed = self.pooler(embeds, attention_mask)
        if self.loss_str == 'infoNCE_euclidean':
            return pooled_embed
        return F.normalize(pooled_embed)

