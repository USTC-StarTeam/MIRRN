# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pandas.core.common import flatten
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import FeatureEmbeddingDict, MLP_Block, DIN_Attention, Dice, MultiHeadTargetAttention
from fuxictr.pytorch.layers import ScaledDotProductAttention
import math


class FilterLayer(nn.Module):
    def __init__(self, max_length, hidden_size, hidden_dropout_prob):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FilterLayer2(nn.Module):
    def __init__(self, max_length, hidden_size, hidden_dropout_prob, n_block):
        super(FilterLayer2, self).__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(n_block, hidden_size // n_block, hidden_size // n_block, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.n = n_block

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape

        A = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        A = A.view(batch, seq_len // 2 + 1, self.n, hidden // self.n)
        B = torch.view_as_complex(self.complex_weight)
        C = torch.einsum("blnd,ndd->blnd", A, B)
        C = C.view(batch, seq_len // 2 + 1, hidden)
        sequence_emb_fft = torch.fft.irfft(C, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class MIRRN(BaseModel):
    def __init__(self,
                 feature_map,
                 model_id="MIRRN",
                 gpu=-1,
                 dnn_hidden_units=[512, 128, 64],
                 dnn_activations="ReLU",
                 attention_dim=64,
                 num_heads=1,
                 use_scale=True,
                 attention_dropout=0,
                 reuse_hash=True,
                 hash_bits=32,
                 topk=50,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 net_dropout=0,
                 batch_norm=False,
                 short_target_field=[("item_id", "cate_id")],
                 short_sequence_field=[("click_history", "cate_history")],
                 long_target_field=[("item_id", "cate_id")],
                 long_sequence_field=[("click_history", "cate_history")],
                 embedding_regularizer=None,
                 net_regularizer=None,
                 **kwargs):
        super(MIRRN, self).__init__(feature_map,
                                    model_id=model_id,
                                    gpu=gpu,
                                    embedding_regularizer=embedding_regularizer,
                                    net_regularizer=net_regularizer,
                                    **kwargs)
        if type(short_target_field) != list:
            short_target_field = [short_target_field]
        if type(short_sequence_field) != list:
            short_sequence_field = [short_sequence_field]
        if type(long_target_field) != list:
            long_target_field = [long_target_field]
        if type(long_sequence_field) != list:
            long_sequence_field = [long_sequence_field]
        self.short_target_field = short_target_field
        self.short_sequence_field = short_sequence_field
        self.long_target_field = long_target_field
        self.long_sequence_field = long_sequence_field
        assert len(self.short_target_field) == len(self.short_sequence_field) \
               and len(self.long_target_field) == len(self.long_sequence_field), \
            "Config error: target_field mismatches with sequence_field."
        self.feature_map = feature_map
        self.embedding_dim = embedding_dim
        # ETA
        self.reuse_hash = reuse_hash
        self.hash_bits = hash_bits
        self.topk = topk
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.short_attention = nn.ModuleList()
        for target_field in self.short_target_field:
            if type(target_field) == tuple:
                input_dim = embedding_dim * len(target_field)
            else:
                input_dim = embedding_dim
            self.short_attention.append(MultiHeadTargetAttention(input_dim,
                                                                 attention_dim,
                                                                 num_heads,
                                                                 attention_dropout,
                                                                 use_scale))
        self.long_attention = nn.ModuleList()
        self.MHFT_block = nn.ModuleList()
        self.random_rotations = nn.ParameterList()
        self.pos = nn.Embedding(300 + 1, embedding_dim)
        for target_field in self.long_target_field:
            if type(target_field) == tuple:
                input_dim = embedding_dim * len(target_field)
            else:
                input_dim = embedding_dim
            self.random_rotations.append(nn.Parameter(torch.randn(input_dim, self.hash_bits),
                                                      requires_grad=False))
            self.MHFT_block.append(FilterLayer2(topk, embedding_dim, 0.1, 4))
            self.MHFT_block.append(FilterLayer2(topk, embedding_dim, 0.1, 4))
            self.MHFT_block.append(FilterLayer2(topk, embedding_dim, 0.1, 4))
            self.long_attention.append(MultiHeadTargetAttention(input_dim,
                                                                attention_dim,
                                                                num_heads,
                                                                attention_dropout,
                                                                use_scale))

        self.dnn = MLP_Block(input_dim=feature_map.sum_emb_out_dim(),
                             output_dim=1,
                             hidden_units=dnn_hidden_units,
                             hidden_activations=dnn_activations,
                             output_activation=self.output_activation,
                             dropout_rates=net_dropout,
                             batch_norm=batch_norm)

        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.reset_parameters()
        self.model_to_device()

    def forward(self, inputs):
        X = self.get_inputs(inputs)
        feature_emb_dict = self.embedding_layer(X)
        # short interest attention
        for idx, (target_field, sequence_field) in enumerate(zip(self.short_target_field,
                                                                 self.short_sequence_field)):
            target_emb = self.concat_embedding(target_field, feature_emb_dict)
            sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0]  # flatten nested list to pick the first field
            mask = X[seq_field].long() != 0  # padding_idx = 0 required in input data
            short_interest_emb = self.short_attention[idx](target_emb, sequence_emb, mask)
            for field, field_emb in zip(list(flatten([sequence_field])),
                                        short_interest_emb.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb

        # long interest attention
        for idx, (target_field, sequence_field) in enumerate(zip(self.long_target_field,
                                                                 self.long_sequence_field)):
            target_emb = self.concat_embedding(target_field, feature_emb_dict)
            sequence_emb = self.concat_embedding(sequence_field, feature_emb_dict)
            seq_field = list(flatten([sequence_field]))[0]  # flatten nested list to pick the first field
            mask = X[seq_field].long() != 0  # padding_idx = 0 required in input data

            # target
            topk_target_emb, topk_target_mask, topk_target_index = self.topk_retrieval(self.random_rotations[idx],
                                                                                       target_emb, sequence_emb, mask,
                                                                                       self.topk)

            # short
            short_emb = sequence_emb[:, -16:]
            mean_short_emb = short_emb.mean(1)
            topk_short_emb, topk_short_mask, topk_short_index = self.topk_retrieval(self.random_rotations[idx],
                                                                                    mean_short_emb, sequence_emb, mask,
                                                                                    self.topk)

            # global
            mean_global_emb = sequence_emb.mean(1)
            topk_global_emb, topk_global_mask, topk_global_index = self.topk_retrieval(self.random_rotations[idx],
                                                                                       mean_global_emb, sequence_emb,
                                                                                       mask, self.topk)

            # pos
            pos_mask_target = sequence_emb.shape[1] - topk_target_index
            pos_target = self.pos(pos_mask_target)
            topk_target_emb += pos_target * 0.02

            pos_mask_short = sequence_emb.shape[1] - topk_short_index
            pos_short = self.pos(pos_mask_short)
            topk_short_emb += pos_short * 0.02

            pos_mask_global = sequence_emb.shape[1] - topk_global_index
            pos_global = self.pos(pos_mask_global)
            topk_global_emb += pos_global * 0.02

            # MHFT
            target_interest_emb = self.MHFT_block[idx * 3](topk_target_emb)
            target_interest_emb = target_interest_emb.mean(1)
            short_interest_emb = self.MHFT_block[idx * 3 + 1](topk_short_emb)
            short_interest_emb = short_interest_emb.mean(1)
            global_interest_emb = self.MHFT_block[idx * 3 + 2](topk_global_emb)
            global_interest_emb = global_interest_emb.mean(1)

            interest_emb = torch.stack((target_interest_emb, short_interest_emb, global_interest_emb), 1)
            interest = self.long_attention[idx](target_emb, interest_emb)

            for field, field_emb in zip(list(flatten([sequence_field])),
                                        interest.split(self.embedding_dim, dim=-1)):
                feature_emb_dict[field] = field_emb

        feature_emb = self.embedding_layer.dict2tensor(feature_emb_dict, flatten_emb=True)
        y_pred = self.dnn(feature_emb)
        return_dict = {"y_pred": y_pred}
        return return_dict

    def concat_embedding(self, field, feature_emb_dict):
        if type(field) == tuple:
            emb_list = [feature_emb_dict[f] for f in field]
            return torch.cat(emb_list, dim=-1)
        else:
            return feature_emb_dict[field]

    def topk_retrieval(self, random_rotations, target_item, history_sequence, mask, topk=5):
        if not self.reuse_hash:
            random_rotations = torch.randn(target_item.size(1), self.hash_bits, device=target_item.device)
        target_hash = self.lsh_hash(target_item.unsqueeze(1), random_rotations)
        sequence_hash = self.lsh_hash(history_sequence, random_rotations)
        hash_sim = -torch.abs(sequence_hash - target_hash).sum(dim=-1)
        hash_sim = hash_sim.masked_fill_(mask.float() == 0, -(self.hash_bits + 1))
        topk_index = hash_sim.topk(topk, dim=1, largest=True, sorted=True)[1]
        topk_index = topk_index.sort(-1)[0]
        topk_emb = torch.gather(history_sequence, 1,
                                topk_index.unsqueeze(-1).expand(-1, -1, history_sequence.shape[-1]))
        topk_mask = torch.gather(mask, 1, topk_index)
        return topk_emb, topk_mask, topk_index

    def lsh_hash(self, vecs, random_rotations):
        """ See the tensorflow-lsh-functions for reference:
            https://github.com/brc7/tensorflow-lsh-functions/blob/main/lsh_functions.py

            Input: vecs, with hape B x seq_len x d
        """
        rotated_vecs = torch.matmul(vecs, random_rotations)  # B x seq_len x num_hashes
        hash_code = torch.relu(torch.sign(rotated_vecs))
        return hash_code


