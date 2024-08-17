import torch
import torch.nn as nn
import math
from torch.autograd import Variable
import numpy as np
import os
import torch.nn.functional as F
import copy
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)  # max_len:句子最大长度 embedding_dim：词嵌入的维度
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2)) * -(math.log(10000.0) / embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        x = self.dropout(x)
        return x


def shape_mask(s1, s2, k=1):
    attn_shape = (1, s1, s2)
    s_mask = np.triu(np.ones(attn_shape), k=k).astype("uint8")
    s_mask = torch.from_numpy(1 - s_mask)
    return s_mask


def softmax_1(scores):
    scores = scores - torch.max(scores, dim=3, keepdim=True)[0]
    X_exp = scores.exp()
    X_sum = X_exp.sum(dim=3, keepdim=True)
    # P = X_exp / (1 + X_sum)
    P = X_exp / X_sum
    return P


class SoftmaxResNet(nn.Module):
    def __init__(self, in_out, d_ff, dropout=0.1):
        super(SoftmaxResNet, self).__init__()
        self.w1 = nn.Linear(in_out, d_ff)
        self.w2 = nn.Linear(d_ff, in_out)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = F.gelu(self.w1(x))
        out = self.dropout(out)
        out = x + self.w2(out)
        return out


def softmax_pow(scores):
    X_exp = scores
    # X_exp = scores.pow(2)
    X_sum = X_exp.sum(dim=3, keepdim=True)
    P = X_exp / X_sum
    return P


def attention(query, key, value, sf_net, mask=None, dropout=None):
    embedding_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embedding_dim)

    scores = sf_net(scores)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # -1e9

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    attn = torch.matmul(p_attn, value)
    return attn, p_attn


def attention_ori(query, key, value, mask=None, dropout=None):
    embedding_dim = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(embedding_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    attn = torch.matmul(p_attn, value)
    return attn, p_attn


def clones(module, N):
    module_list = nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    return module_list


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, embedding_dim, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % heads == 0, "词嵌入维度无法整除head数"
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.p_attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.sf_net = copy.deepcopy(SoftmaxResNet(71, 128))

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = [model(x).view(batch_size, -1, self.heads, self.embedding_dim // self.heads).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        x, self.p_attn = attention(query, key, value, self.sf_net, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        x = self.linears[-1](x)

        return x


class MultiHeadedAttentionOri(nn.Module):
    def __init__(self, heads, embedding_dim, dropout=0.1):
        super(MultiHeadedAttentionOri, self).__init__()
        assert embedding_dim % heads == 0, "词嵌入维度无法整除head数"
        self.heads = heads
        self.embedding_dim = embedding_dim
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.p_attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        query, key, value = [model(x).view(batch_size, -1, self.heads, self.embedding_dim // self.heads).transpose(1, 2)
                             for model, x in zip(self.linears, (query, key, value))]

        x, self.p_attn = attention_ori(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_dim)
        x = self.linears[-1](x)

        return x


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(embedding_dim, d_ff)
        self.w2 = nn.Linear(d_ff, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.gelu(self.w1(x))
        x = self.dropout(x)
        x = self.w2(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a2 = nn.Parameter(torch.ones(embedding_dim))
        self.b2 = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a2 * (x - mean) / (std + self.eps) + self.b2
        return x


class SublayerConnection(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        print(x.shape)
        print(sublayer(self.norm(x)).shape)
        x = x + self.dropout(sublayer(self.norm(x)))
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(embedding_dim, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda f: self.self_attn(x, x, x, mask))
        x = self.sublayers[1](x, lambda f: self.feed_forward(x))  # 与此等价：x = self.sublayers[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):
        super(Encoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, N)
        self.norm = LayerNorm(encoder_layer.embedding_dim)

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
            # print(x)
        x = self.norm(x)
        return x


class MakeTransformer(nn.Module):
    def __init__(self, max_len, e_dim, d_ff, heads, N, out_shape, dropout):
        super(MakeTransformer, self).__init__()
        self.PositionalEnconding_net = PositionalEncoding(embedding_dim=e_dim, max_len=max_len,
                                                          dropout=0)

        self.ff = PositionWiseFeedForward(embedding_dim=e_dim, d_ff=d_ff, dropout=dropout)
        self.self_attn = MultiHeadedAttention(heads=heads, embedding_dim=e_dim, dropout=dropout)

        self.EncoderLayer_net = EncoderLayer(embedding_dim=e_dim, self_attn=copy.deepcopy(self.self_attn),
                                             feed_forward=copy.deepcopy(self.ff), dropout=dropout)

        self.Encoder_net = copy.deepcopy(Encoder(encoder_layer=copy.deepcopy(self.EncoderLayer_net), N=N))
        self.linear = nn.Linear(e_dim, out_shape)

    def forward(self, x, mask):
        print(x.shape)
        x = self.PositionalEnconding_net(x)
        out = self.Encoder_net(x, mask)
        out = self.linear(F.gelu(out))
        return out


class MakeTransformerOri(nn.Module):
    def __init__(self, max_len, e_dim, d_ff, heads, N, out_shape, dropout):
        super(MakeTransformerOri, self).__init__()
        self.PositionalEnconding_net = PositionalEncoding(embedding_dim=e_dim, max_len=max_len,
                                                          dropout=0)

        self.ff = PositionWiseFeedForward(embedding_dim=e_dim, d_ff=d_ff, dropout=dropout)
        self.self_attn = MultiHeadedAttentionOri(heads=heads, embedding_dim=e_dim, dropout=dropout)

        self.EncoderLayer_net = EncoderLayer(embedding_dim=e_dim, self_attn=copy.deepcopy(self.self_attn),
                                             feed_forward=copy.deepcopy(self.ff), dropout=dropout)

        self.Encoder_net = copy.deepcopy(Encoder(encoder_layer=copy.deepcopy(self.EncoderLayer_net), N=N))
        self.linear = nn.Linear(e_dim, out_shape)

    def forward(self, x, mask):
        x = self.PositionalEnconding_net(x)
        out = self.Encoder_net(x, mask)
        out = self.linear(F.gelu(out))
        return out


def get_data_(data_name, NP="N", sk_id=None):
    if sk_id is None:
        sk_id = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    if NP == "N":
        L = [[1, 0]]  # [1, 0]是正常人
    elif NP == "P":
        L = [[0, 1]]  # [0, 1]是帕金森
    elif NP == "A":
        L = [[1, 0], [0, 1]]
    else:
        raise Exception('NP指定"N"、"P"、"A')

    data_ = np.load(data_name, allow_pickle=True)
    print(data_.shape)
    features_ = []
    label_ = []
    for i in range(len(data_)):
        if data_[i][1] in L:
            label_.append((data_[i][1]))
            data_[i][0] = data_[i][0][:, sk_id, :]
            features_.append(np.array(torch.from_numpy(data_[i][0]).transpose(-2, -1)))

    features_ = torch.as_tensor(np.array(features_), dtype=torch.float32)
    features_ = features_.view(-1, features_.shape[1], features_.shape[2] * features_.shape[3])

    label_ = torch.as_tensor(np.array(label_), dtype=torch.float32)

    return features_, label_


def model_load(net, mask, dl, lf, lr_w, load_=True, path="./model_save/t.pth"):
    if os.path.exists(path):
        if load_:
            net.load_state_dict(torch.load(path))
            print("读取模型：" + path)
            return
    else:
        for p in net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                print("初始化模型：" + path)
                print("进行warmup")
                warm_up(net, mask, dl, lf, lr_end=lr_w)
                return


def model_load_P(net, mask, dl, lf, lr_w, load_=True, path="./model_save/t.pth"):
    if os.path.exists(path):
        if load_:
            net.load_state_dict(torch.load(path))
            print("读取模型：" + path)
            return
    else:
        for p in net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                print("初始化模型：" + path)
                print("进行warmup")
                warm_up_P(net, mask, dl, lf, lr_end=lr_w)
                return


def model_load_eval(net, path="./model_save/t.pth"):
    net.load_state_dict(torch.load(path))
    print("读取模型：" + path)
    return


def model_load_C(net, load_=True, path="./model_save/t.pth"):
    if os.path.exists(path):
        if load_:
            net.load_state_dict(torch.load(path))
            print("读取模型：" + path)
            return
    else:
        for p in net.parameters():
            if p.dim() > 1:
                print("初始化模型：" + path)
                nn.init.xavier_uniform_(p)
                return


def model_save(argument, name, net, max_min, argument_path="model_save/t.csv", net_path="model_save/t.pth"):
    if not os.path.exists(argument_path):
        if max_min == "max":
            m = 0.0
        elif max_min == "min":
            m = 1000.0
        else:
            raise Exception('指定"max"或者"min"')
        a = pd.DataFrame([m], columns=[name])
        a.index.name = "index"
        a.to_csv(argument_path)
    else:
        argument_m = pd.read_csv(argument_path)
        if max_min == "max":
            if argument >= argument_m.loc[0, name]:
                if argument > argument_m.loc[0, name]:
                    print("已更新最佳模型:", argument)
                    argument_m.loc[0, name] = argument
                    argument_m.to_csv(argument_path, index=False)
                    torch.save(net.state_dict(), net_path)
                    return
        elif max_min == "min":
            if argument <= argument_m.loc[0, name]:
                if argument < argument_m.loc[0, name]:
                    print("已更新最佳模型," + name + ":", argument)
                    argument_m.loc[0, name] = argument
                    argument_m.to_csv(argument_path, index=False)
                    torch.save(net.state_dict(), net_path)
                    return


def warm_up(net, mask, data_load, loss_f, lr_end=1e-3):
    lr = 1e-12
    for epoch in range(10000):
        batch_loss = []
        opt = torch.optim.RAdam(net.parameters(), betas=(0.9, 0.99), lr=lr)
        for x, y in data_load:
            net.train()
            out = net(x, mask)
            loss = loss_f(out, y)
            batch_loss.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            if lr >= lr_end:
                print("warmup end", 'loss =', '{:.9f}'.format(np.mean(batch_loss)))
                break
            print("warm_up Epoch:", '%02d' % (epoch // 10), 'loss =', '{:.9f}'.format(np.mean(batch_loss)))
            lr *= 10


def warm_up_P(net, mask, data_load, loss_f, lr_end=1e-3):
    lr = 1e-12
    for epoch in range(10000):
        batch_loss = []
        opt = torch.optim.RAdam(net.parameters(), betas=(0.9, 0.99), lr=lr)
        for x, y, z in data_load:
            net.train()
            out = net(torch.cat([x, y], dim=2), mask)
            loss = loss_f(out, y)
            batch_loss.append(loss.detach().cpu().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            if lr >= lr_end:
                print("warmup end", 'loss =', '{:.9f}'.format(np.mean(batch_loss)))
                break
            print("warm_up Epoch:", '%02d' % (epoch // 10), 'loss =', '{:.9f}'.format(np.mean(batch_loss)))
            lr *= 10


class ClassificationNet71(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(71, 128, (1, 1), (1, 1))
        self.dp = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 2 * 13, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x1):
        out = self.dp(F.gelu(self.layer1(x1)))
        out = out.view(-1, 128 * 2 * 13)
        out = self.dp(F.gelu(self.fc1(out)))
        out = F.softmax(self.fc2(out), dim=1)
        return out


class ClassificationNet72(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(72, 128, (1, 1), (1, 1))
        self.dp = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 2 * 13, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x1):
        out = self.dp(F.gelu(self.layer1(x1)))
        out = out.view(-1, 128 * 2 * 13)
        out = self.dp(F.gelu(self.fc1(out)))
        out = F.softmax(self.fc2(out), dim=1)
        return out


class ClassificationNetAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(71, 128, (1, 1), (1, 1))
        self.layer2 = nn.Conv2d(72, 128, (1, 1), (1, 1))
        self.dp = nn.Dropout(0.1)
        self.fc1 = nn.Linear(256 * 2 * 13, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x1, x2):
        out1 = self.dp(F.gelu(self.layer1(x1)))
        out2 = self.dp(F.gelu(self.layer2(x2)))
        out = torch.cat([out1, out2], dim=1)
        out = out.view(-1, 256 * 2 * 13)
        out = self.dp(F.gelu(self.fc1(out)))
        out = F.softmax(self.fc2(out), dim=1)
        return out


class ClassificationNet71A(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(71 * 2 * 13, 2)

    def forward(self, x1):
        out = x1.view(-1, 71 * 2 * 13)
        out = F.softmax(self.fc1(out), dim=1)
        return out


class ClassificationNet72A(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(72 * 2 * 13, 2)

    def forward(self, x1):
        out = x1.view(-1, 72 * 2 * 13)
        out = F.softmax(self.fc1(out), dim=1)
        return out


class ClassificationNetAddA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(143 * 2 * 13, 2)

    def forward(self, x1, x2):
        out1 = x1.view(-1, 71 * 2 * 13)
        out2 = x2.view(-1, 72 * 2 * 13)
        out = torch.cat([out1, out2], dim=1)
        out = F.softmax(self.fc1(out), dim=1)
        return out
