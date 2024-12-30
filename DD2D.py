"""
######
DD2D model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
#######
"""

import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)



class TransformerConfig:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class CausalSelfAttention_masked_for_crystal(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, is_crystal=None):
        B, T, C = x.size()

        x_padding_judge = 1.0 - x_padding_judge
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=2)
        x_padding_judge = x_padding_judge @ x_padding_judge.transpose(-2, -1)
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=1)
        x_padding_judge = torch.tile(x_padding_judge, [1, self.n_head, 1, 1])

        if is_crystal:
            x_padding_judge[:, :, 0, 7:] = 0.0

        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att_full = torch.ones_like(att)
        att_full = self.attn_drop(att_full)
        x_padding_judge = x_padding_judge * att_full
        att = att.masked_fill(x_padding_judge == 0, -1e9)
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_drop(self.proj(y))
        return y


class Transformer_encoder_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn1 = CausalSelfAttention_masked_for_crystal(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, x_padding_judge, is_crystal):
        x = x + self.attn1(self.ln1(x), x_padding_judge, is_crystal)
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer_pattern_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge):
        for block in self.blocks:
            x = block(x, x_padding_judge, is_crystal=False)
        return x


class Transformer_crystal_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge):
        for block in self.blocks:
            x = block(x, x_padding_judge, is_crystal=True)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


class pattern_emb(nn.Module):
    def __init__(self, config, in_channels=3):
        super(pattern_emb, self).__init__()
        emb_every = config.n_embd // in_channels
        emb_remainder = config.n_embd % in_channels
        self.fc1 = nn.Linear(1, emb_every)
        self.fc2 = nn.Linear(1, emb_every)
        self.fc3 = nn.Linear(1, emb_every + emb_remainder)

    def forward(self, xyz):
        xyz = xyz.transpose(dim0=1, dim1=2)
        out1 = self.fc1(xyz[:,:,0:1])
        out2 = self.fc2(xyz[:,:,1:2])
        out3 = self.fc3(xyz[:,:,2:3])
        pattern_emb = torch.cat([out1, out2, out3], dim=2)
        return pattern_emb


class seq_emb(nn.Module):
    def __init__(self, config):
        super(seq_emb, self).__init__()
        self.fc1 = nn.Linear(1, config.n_embd)

    def forward(self, seq):
        seq = seq.unsqueeze(dim=2)
        seq_emb = self.fc1(seq)
        return seq_emb


class project_mlp(nn.Module):
    def __init__(self, config):
        super(project_mlp, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)


class project_mlp_crystal(nn.Module):
    def __init__(self, config):
        super(project_mlp_crystal, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)


class DD2D(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        embeddingSize = config.n_embd

        self.block_size = config.block_size

        self.tok_emb = seq_emb(config)
        self.pos_emb = PositionalEncoding(embeddingSize, dropout=0.1, max_len=config.block_size)

        self.pattern_emb = pattern_emb(config, in_channels=3)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.fc_project_crystal = project_mlp_crystal(config)
        self.fc_project_pattern = project_mlp(config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.register_buffer('total_labels', torch.arange(10000))

        self.Trans_crystal_encoder = Transformer_crystal_encoder(config)
        self.Trans_pattern_encoder = Transformer_pattern_encoder(config)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.ln_p = nn.LayerNorm(config.n_embd)

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('logit_scale')

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, crystal, pattern=None, variables=None):
        b, t = crystal.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        crystal_mask = crystal == -1.0
        crystal_mask = crystal_mask.float()

        token_embeddings = self.tok_emb(crystal)  
        input_embedding = self.pos_emb(token_embeddings)  

        pattern_judge = torch.norm(pattern, p=2, keepdim=False, dim=1)
        pattern_judge = pattern_judge == 0.0
        pattern_judge = pattern_judge.float()
        pattern_embeddings = self.pattern_emb(pattern)

        crystal_embedding = self.drop(input_embedding)
        crystal_embedding = self.Trans_crystal_encoder(crystal_embedding, crystal_mask)
        crystal_embedding = self.ln_f(crystal_embedding)

        crystal_embedding_final = crystal_embedding[torch.arange(crystal_embedding.shape[0]), crystal.argmax(dim=-1)]\
              + crystal_embedding[torch.arange(crystal_embedding.shape[0]), 0]
        
        pattern_embeddings = self.Trans_pattern_encoder(pattern_embeddings, pattern_judge)
        pattern_embeddings = self.ln_p(pattern_embeddings)

        crystal_embedding_final = self.fc_project_crystal(crystal_embedding_final)
        pattern_embeddings_final = pattern_embeddings[torch.arange(pattern_embeddings.shape[0]), pattern[:,2,:].argmax(dim=-1)]
        pattern_embeddings_final = self.fc_project_pattern(pattern_embeddings_final)

        crystal_embedding_final = crystal_embedding_final / crystal_embedding_final.norm(dim=-1, keepdim=True)
        pattern_embeddings_final = pattern_embeddings_final / pattern_embeddings_final.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_pattern = logit_scale * pattern_embeddings_final @ crystal_embedding_final.t()
        logits_per_crystal = logits_per_pattern.t()

        labels = self.total_labels[:b]
        loss = (F.cross_entropy(logits_per_pattern, labels) +
                F.cross_entropy(logits_per_crystal, labels)) / 2

        return loss, logits_per_pattern
<<<<<<< HEAD
=======

>>>>>>> de8745369266226120f07ea03742679ab771f4bf
