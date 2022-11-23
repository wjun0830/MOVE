import torch
from torch import nn
from .net_module import NetVLAGD
import torch.nn.functional as F
import numpy as np
import random

class NonlinearClassifier(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        super(NonlinearClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 2048)
        self.relu = nn.ReLU()
        self.project = nn.Linear(2048, num_class)
        self.drop_layer = nn.Dropout(p=0.5)
        self.early = False

    def forward(self, x):
        logits = self.logits(x)
        prediction = torch.sigmoid(logits)
        return prediction, logits

    def logits(self, x):
        if self.early:
            x = torch.mean(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop_layer(x)
        logits = self.project(x)

        if not self.early:
            logits = torch.mean(logits, 1)
        return logits

class NetVLAG_Dynamic_AGG(nn.Module):
    def __init__(self, feature_size, cluster_size, add_batch_norm, add_bias=False, gating=True):
        super(NetVLAG_Dynamic_AGG, self).__init__()
        self.feature_size = feature_size
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

        self.add_bias = add_bias
        self.gating = gating

        self.activation_fc = nn.Linear(feature_size, cluster_size, bias=add_bias)
        self.gate_weight = nn.Parameter(torch.zeros(cluster_size, feature_size))
        self.gate_weight = nn.init.xavier_normal_(self.gate_weight)

        if add_batch_norm:
            self.bn1 = nn.BatchNorm1d(cluster_size)

    def forward(self, x, sampling_prob=None, label=None, train=True):
        x1 = x.detach().clone()
        x2 = x.detach().clone()
        batch_size, max_frames, feature_size = x.shape
        if train:
            target = [label]
            for tidx, tar in enumerate(target):
                lower_query_sampling_num = np.mean(sampling_prob[tar]).astype(np.uint32)

                query_sampling_num = random.randint(lower_query_sampling_num, max_frames)
                query_sampling_idx = list(set(list(range(max_frames))) - set(random.sample(range(max_frames), query_sampling_num)))
                x1[tidx, query_sampling_idx] = x1[tidx, query_sampling_idx] * 0

                query_sampling_num2 = random.randint(lower_query_sampling_num, max_frames)
                query_sampling_idx2 = list(set(list(range(max_frames))) - set(random.sample(range(max_frames), query_sampling_num2)))
                x2[tidx, query_sampling_idx2] = x2[tidx, query_sampling_idx2] * 0

            x = torch.cat([x1, x2], dim=0)
        batch_size, max_frames, feature_size = x.shape
        x = x.view(batch_size * max_frames, feature_size)
        activation = self.activation_fc(x)
        if self.add_batch_norm:
            activation = self.bn1(activation)

        activation = F.softmax(activation, 1)
        activation = activation.view(-1, max_frames, self.cluster_size)
        activation = activation.permute(0, 2, 1)
        reshaped_input = x.view(-1, max_frames, feature_size)

        vlagd = torch.matmul(activation, reshaped_input)

        if self.gating:
            gate_weight = F.sigmoid(self.gate_weight)
            vlagd = torch.mul(vlagd, gate_weight)
            vlagd = vlagd.permute(0, 2, 1)

        vlagd = F.normalize(vlagd, p=2, dim=1)
        vlagd = vlagd.reshape(batch_size, self.cluster_size * feature_size)
        vlagd = F.normalize(vlagd, p=2, dim=1)
        return vlagd

class NetVLAD_Dynamic_Aggregator(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        # feature extractor
        super(NetVLAD_Dynamic_Aggregator, self).__init__()

        # basic params setting
        self.feature_size = feature_dim

        cluster_size = 64
        hidden_size = 1024
        self.add_batch_norm = True
        self.relu = True

        self.vlad = NetVLAG_Dynamic_AGG(feature_size=self.feature_size,
                             cluster_size=cluster_size,
                             add_batch_norm=True,
                             add_bias=False)

        # classifier
        self.hidden_fc = nn.Linear(cluster_size * self.feature_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU6()

    def forward(self, x, sampling_prob=None, label=None, train=True):

        batch_size, max_frames, feature_size = x.shape

        x = x.view(batch_size * max_frames, feature_size)
        if self.add_batch_norm:
            x = self.bn1(x)
            x = x.view(batch_size, max_frames, feature_size)
        vlad = self.vlad(x, sampling_prob, label, train)
        # activation -> [bs, hidden_size]
        activation = self.hidden_fc(vlad)
        if self.add_batch_norm and self.relu:
            activation = self.bn2(activation)

        if self.relu:
            activation = self.relu1(activation)
        hbsz = len(activation) // 2
        if train:
            return activation[:hbsz], activation[hbsz:]
        else:
            return activation



class MultiHeadAttention_Dynamic_Aggregator(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head=4, d_model=2048, d_k=512, d_v=512, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, sampling_prob=None, label=None, train=True, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        if train:
            target = [label]
            new_q = q[:, :2, :].detach().clone()
            mask = torch.ones(sz_b, 1, 60).cuda()
            for tidx, tar in enumerate(target):
                lower_query_sampling_num = np.mean(sampling_prob[tar]).astype(np.uint32)
                for ii in range(2):
                    query_sampling_num = random.randint(lower_query_sampling_num, len_q)
                    query_sampling_idx = random.sample(range(len_q), query_sampling_num)
                    new_q[tidx, ii] = q[tidx, query_sampling_idx].mean(0)
                query_sampling_num2 = random.randint(lower_query_sampling_num, len_q)
                query_sampling_idx2 = random.sample(range(len_q), query_sampling_num2)
                mask[tidx, 0, query_sampling_idx2] = 0
            residual = new_q # 128, 2, 2048
            len_q = 2
        else:
            bsz, nframe, fdim = q.shape
            new_q = q.mean(1).unsqueeze(1)
            residual = new_q
            len_q = 1
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        new_q = self.w_qs(new_q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        new_q, k, v = new_q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        new_q, attn = self.attention(new_q, k, v, mask=mask)
        # 128        4        1        512
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        new_q = new_q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        new_q = self.dropout(self.fc(new_q))
        new_q += residual
        new_q = self.layer_norm(new_q)
        return new_q, attn

class NetVLADModel_Aggregator(nn.Module):
    def __init__(self, feature_dim=2048, num_class=1004):
        # feature extractor
        super(NetVLADModel_Aggregator, self).__init__()

        # basic params setting
        self.feature_size = feature_dim

        cluster_size = 64
        hidden_size = 1024
        self.add_batch_norm = True
        self.relu = True

        self.vlad = NetVLAGD(feature_size=self.feature_size,
                             cluster_size=cluster_size,
                             add_batch_norm=True,
                             add_bias=False)

        # classifier
        self.hidden_fc = nn.Linear(cluster_size * self.feature_size, hidden_size, bias=False)
        self.bn1 = nn.BatchNorm1d(self.feature_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.relu1 = nn.ReLU6()

    def forward(self, x):
        logits = self.logits(x)
        return logits

    def logits(self, x):
        batch_size, max_frames, feature_size = x.shape

        x = x.view(batch_size * max_frames, feature_size)

        if self.add_batch_norm:
            x = self.bn1(x)
            x = x.view(batch_size, max_frames, feature_size)

        vlad = self.vlad(x)
        activation = self.hidden_fc(vlad)
        if self.add_batch_norm and self.relu:
            activation = self.bn2(activation)

        if self.relu:
            activation = self.relu1(activation)

        return activation

class MultiHeadAttention_Aggregator(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head=4, d_model=2048, d_k=512, d_v=512, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, train=True, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1) # 128 60 2048
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        bsz, nframe, fdim = q.shape
        q = q.mean(1).unsqueeze(1)
        residual = q
        len_q = q.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # 128 60 4 512
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


