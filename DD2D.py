"""
DD2D model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, block_size, **kwargs):
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention_masked_for_formula(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, x_padding_judge, is_formula=None):
        B, T, C = x.size()

        x_padding_judge = 1.0 - x_padding_judge
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=2)
        x_padding_judge = x_padding_judge @ x_padding_judge.transpose(-2, -1)
        x_padding_judge = torch.unsqueeze(x_padding_judge, dim=1)
        x_padding_judge = torch.tile(x_padding_judge, [1, self.n_head, 1, 1])

        # if is_formula:
        #     x_padding_judge[:, :, 0, 7:] = 0.0

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        ##########Dropkey#####################
        att_full = torch.ones_like(att)
        att_full = self.attn_drop(att_full)
        x_padding_judge = x_padding_judge * att_full
        ############################################
        att = att.masked_fill(x_padding_judge == 0, -1e9)
        att = F.softmax(att, dim=-1)
        # att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y



class Transformer_encoder_block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn1 = CausalSelfAttention_masked_for_formula(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, x_padding_judge, is_formula):
        x = x + self.attn1(self.ln1(x), x_padding_judge, is_formula)
        x = x + self.mlp(self.ln2(x))
        return x




class Transformer_point_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge):
        for block in self.blocks:
            x = block(x, x_padding_judge, is_formula=False)
        return x







class Transformer_formula_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([Transformer_encoder_block(config) for _ in range(config.n_layer)])

    def forward(self, x, x_padding_judge):
        for block in self.blocks:
            x = block(x, x_padding_judge, is_formula=True)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        """
        :param d_model: pe encoding dimension, usually the same as word embedding, for easy addition
        :param dropout: dorp out
        :param max_len: the length of the longest sentence in the corpus, i.e. L in word embedding
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，
        self.register_buffer('pe', pe)  # 

    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)  # size = [batch, L, d_model]
        return x  # size = [batch, L, d_model]


class PointNetConfig:
    """ base PointNet config """

    def __init__(self, embeddingSize, numberofPoints, numberofVars,
                 numberofYs, method='GPT', varibleEmbedding='NOT_VAR',
                 **kwargs):
        self.embeddingSize = embeddingSize
        self.numberofPoints = numberofPoints  # number of points
        self.numberofVars = numberofVars  # input dimension (Xs)
        self.numberofYs = numberofYs  # output dimension (Ys)
        self.method = method
        self.varibleEmbedding = varibleEmbedding

        for k, v in kwargs.items():
            setattr(self, k, v)





# pointNet based on Convolution, T-NET naming is not accurate
class tNet(nn.Module):
    """
    The PointNet structure in the orginal PointNet paper:
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation by Qi et. al. 2017
    """
    def __init__(self, config):
        super(tNet, self).__init__()

        self.activation_func = F.relu
        self.num_units = config.embeddingSize

        self.conv1 = nn.Conv1d(config.numberofVars+config.numberofYs, self.num_units, 1)
        self.conv2 = nn.Conv1d(self.num_units, 2 * self.num_units, 1)
        self.conv3 = nn.Conv1d(2 * self.num_units, 4 * self.num_units, 1)
        self.fc1 = nn.Linear(4 * self.num_units, 2 * self.num_units)
        self.fc2 = nn.Linear(2 * self.num_units, self.num_units)

        #self.relu = nn.ReLU()

        self.input_batch_norm = nn.BatchNorm1d(config.numberofVars+config.numberofYs)
        #self.input_layer_norm = nn.LayerNorm(config.numberofPoints)

        self.bn1 = nn.BatchNorm1d(self.num_units)
        self.bn2 = nn.BatchNorm1d(2 * self.num_units)
        self.bn3 = nn.BatchNorm1d(4 * self.num_units)
        self.bn4 = nn.BatchNorm1d(2 * self.num_units)
        self.bn5 = nn.BatchNorm1d(self.num_units)

    def forward(self, x):
        """
        :param x: [batch, #features, #points]
        :return:
            logit: [batch, embedding_size]
        """
        x = self.input_batch_norm(x)
        x = self.activation_func(self.bn1(self.conv1(x)))
        x = self.activation_func(self.bn2(self.conv2(x)))
        x = self.activation_func(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, dim=2)  # global max pooling
        assert x.size(1) == 4 * self.num_units

        x = self.activation_func(self.bn4(self.fc1(x)))
        x = self.activation_func(self.bn5(self.fc2(x)))
        #x = self.fc2(x)

        return x




class points_emb(nn.Module):
    def __init__(self, config, in_channels=3):
        super(points_emb, self).__init__()
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
        points_emb = torch.cat([out1, out2, out3], dim=2)
        return points_emb



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



class project_mlp_formula(nn.Module):
    def __init__(self, config):
        super(project_mlp_formula, self).__init__()
        self.fc1 = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.fc2 = nn.Linear(2 * config.n_embd, config.n_embd)
        self.act_fun = nn.GELU()
        self.drop = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(self.act_fun(x))
        return self.drop(x)






class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, pointNetConfig=None):
        super().__init__()

        self.config = config
        self.pointNetConfig = pointNetConfig
        self.pointNet = None

        embeddingSize = config.n_embd
        if self.pointNetConfig is not None:

            if self.pointNetConfig.method == 'EMB_CAT':
                print('The model is going to concatenate the embeddings!')
                embeddingSize = config.n_embd // 2  # if concatenation

            # OVERRIDE: POINT embedding should have the same size of token and position embedding
            if self.pointNetConfig.embeddingSize != embeddingSize:
                print("We've override your choice for pointNet embedding! Updating {} with {}!".format(
                    self.pointNetConfig.embeddingSize, embeddingSize))
                self.pointNetConfig.embeddingSize = embeddingSize


            # self.pointNet = PointNet(self.pointNetConfig)

            self.vars_emb = nn.Embedding(self.pointNetConfig.numberofVars + 1, embeddingSize)

            # this is a function with the goal to help the model to converge faster based
            # on the intuitation that given equation it is possible to infer points
            # self.pointFeatures = (self.pointNetConfig.numberofVars+self.pointNetConfig.numberofYs)*self.pointNetConfig.numberofPoints
            # self.helper_batch_norm = nn.BatchNorm1d(self.pointNetConfig.numberofVars+self.pointNetConfig.numberofYs)
            # self.helper = nn.Linear(config.n_embd,
            #         self.pointFeatures,
            #         bias=False)

        if self.pointNetConfig.method == 'EMB_CON':
            print('Add one to the supported block size!')
            self.block_size = config.block_size + 1  # add a first token
            config.block_size += 1
        else:
            self.block_size = config.block_size

        # input embedding stem
        # self.tok_emb = nn.Embedding(config.vocab_size, embeddingSize, padding_idx=self.config.padding_idx)
        self.tok_emb = seq_emb(config)
        # self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, embeddingSize))
        self.pos_emb = PositionalEncoding(embeddingSize, dropout=0.1, max_len=config.block_size)

        self.points_emb = points_emb(config, in_channels=3)

        self.drop = nn.Dropout(config.embd_pdrop)

        self.pointNet = tNet(self.pointNetConfig)
        # self.fc_project_formula = torch.nn.Linear(config.n_embd, config.n_embd)
        # self.fc_project_points = torch.nn.Linear(config.n_embd, config.n_embd)
        self.fc_project_formula = project_mlp_formula(config)
        self.fc_project_points = project_mlp(config)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.register_buffer('total_labels', torch.arange(10000))

        # transformer
        # self.blocks_unmask = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.Trans_formula_encoder = Transformer_formula_encoder(config)
        self.Trans_points_encoder = Transformer_point_encoder(config)
        # self.block = Block(config)
        # decoder head
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
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('logit_scale')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, points=None, variables=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        formula_mask = idx == -1.0
        formula_mask = formula_mask.float()
        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector -> b x length x embedding
        input_embedding = self.pos_emb(token_embeddings)  # [:, :t, :] # each position maps to a (learnable) vector

        if points != None and self.pointNet != None:
            # Total Transformer model
            # points_embeddings = self.pointNet(points)
            points_judge = torch.norm(points, p=2, keepdim=False, dim=1)
            points_judge = points_judge == 0.0
            points_judge = points_judge.float()
            points_embeddings = self.points_emb(points)


            if variables != None and self.pointNetConfig.varibleEmbedding == 'LEA_EMB':
                # add the variables information to the point embedding
                variables_embeddings = self.vars_emb(variables)
                points_embeddings += variables_embeddings



        formula_embedding = self.drop(input_embedding)
        x = self.Trans_formula_encoder(formula_embedding, formula_mask)
        x = self.ln_f(x)
        # formula_embedding_final = x[torch.arange(x.shape[0]), idx.argmax(dim=-1)]
        # formula_embedding_final = x[torch.arange(x.shape[0]), idx.argmax(dim=-1)] + x[torch.arange(x.shape[0]), 0]
        formula_embedding_final = x[torch.arange(x.shape[0]), idx.argmax(dim=-1)] + x[torch.arange(x.shape[0]), 0]
        points_embeddings = self.Trans_points_encoder(points_embeddings, points_judge)
        points_embeddings = self.ln_p(points_embeddings)

        formula_embedding_final = self.fc_project_formula(formula_embedding_final)
        # points_embeddings_final = self.pointNet(points)
        points_embeddings_final = points_embeddings[torch.arange(points_embeddings.shape[0]), points[:,2,:].argmax(dim=-1)]
        points_embeddings_final = self.fc_project_points(points_embeddings_final)

        formula_embedding_final = formula_embedding_final / formula_embedding_final.norm(dim=-1, keepdim=True)
        points_embeddings_final = points_embeddings_final / points_embeddings_final.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_points = logit_scale * points_embeddings_final @ formula_embedding_final.t()
        logits_per_formula = logits_per_points.t()


        labels = self.total_labels[:b]
        loss = (F.cross_entropy(logits_per_points, labels) +
                F.cross_entropy(logits_per_formula, labels)) / 2


        return loss, logits_per_points#, formula_embedding_final
