#%%
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import random
import sys
import time
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
# 打印当前 Python 版本信息
# print(sys.version_info)

# 打印使用库的版本信息
# for module in mpl, np, pd, sklearn, torch :
#     print(module.__name__, module.__version__)

# 设置 PyTorch 计算设备
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# print(device)

# 设置随机种子，以保证实验的可复现性
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
#%% md
# ## DataLoader准备
# ### LangPairDataset
#%%
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
dataset_path = "./train_data_size_1000000"

class LangPairDataset(Dataset):
    """
    加载和处理双语数据集，并支持数据缓存。
    """
    def __init__(self, mode="train", max_length=128, overwrite_cache=False, data_dir=dataset_path):
        """
        初始化数据集。
        :param mode: 数据集模式（"train" 或 "val"）
        :param max_length: 句子最大长度，超过则过滤
        :param overwrite_cache: 是否覆盖缓存，默认为 False
        :param data_dir: 数据目录
        """
        self.data_dir = Path(data_dir)  # 数据存储路径
        cache_path = self.data_dir / ".cache" / f"de2en_{mode}_{max_length}.npy"  # 缓存路径

        if overwrite_cache or not cache_path.exists():  # 覆盖缓存或缓存不存在时重新处理
            cache_path.parent.mkdir(parents=True, exist_ok=True)  # 创建缓存目录

            # 读取源语言和目标语言文件
            with open(f"./{self.data_dir}/{mode}_src.bpe", "r", encoding="utf8") as file:
                self.src = file.readlines()

            with open(f"./{self.data_dir}/{mode}_trg.bpe", "r", encoding="utf8") as file:
                self.trg = file.readlines()

            filtered_src, filtered_trg = [], []  # 存放过滤后的句子

            # 过滤句子长度
            for src, trg in zip(self.src, self.trg):
                if len(src) <= max_length and len(trg) <= max_length:
                    filtered_src.append(src.strip())  # 去除首尾空格
                    filtered_trg.append(trg.strip())

            # 保存为 NumPy 数组并缓存
            np.save(cache_path, {"src": np.array(filtered_src), "trg": np.array(filtered_trg)}, allow_pickle=True)
            print(f"save cache to {cache_path}")

        else:  # 加载已有缓存
            cache_dict = np.load(cache_path, allow_pickle=True).item()
            print(f"load {mode} dataset from {cache_path}")
            filtered_src = cache_dict["src"]
            filtered_trg = cache_dict["trg"]

        self.src = filtered_src  # 源语言数据
        self.trg = filtered_trg  # 目标语言数据

    def __getitem__(self, index):
        """
        获取指定索引的源语言和目标语言句子。
        """
        return self.src[index], self.trg[index]

    def __len__(self):
        """
        返回数据集大小。
        """
        return len(self.src)

# 创建训练集和验证集对象
# train_ds = LangPairDataset("train")
# val_ds = LangPairDataset("val")
#%% md
#  ### Tokenizer
#%%
# 构建英文和中文的word2idx和idx2word字典
en_word2idx = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[UNK]": 2,
    "[EOS]": 3,
}
zh_word2idx = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[UNK]": 2,
    "[EOS]": 3,
}

# 反向索引
en_idx2word = {value: key for key, value in en_word2idx.items()}
zh_idx2word = {value: key for key, value in zh_word2idx.items()}

# 分别加载英文和中文词表
en_index = len(en_idx2word)
zh_index = len(zh_idx2word)
threshold = 1

# 读取英文词表
with open(f"{dataset_path}/en.vocab", "r", encoding="utf8") as file:
    for line in tqdm(file.readlines()):
        token, counts = line.strip().split()
        if int(counts) >= threshold:
            en_word2idx[token] = en_index
            en_idx2word[en_index] = token
            en_index += 1

# 读取中文词表
with open(f"{dataset_path}/zh.vocab", "r", encoding="utf8") as file:
    for line in tqdm(file.readlines()):
        token, counts = line.strip().split()
        if int(counts) >= threshold:
            zh_word2idx[token] = zh_index
            zh_idx2word[zh_index] = token
            zh_index += 1
#%%
class Tokenizer:
    def __init__(self, word2idx, idx2word, max_length=128, pad_idx=0, bos_idx=1, eos_idx=3, unk_idx=2):
        """
        初始化 Tokenizer。
        :param word2idx: 单词到索引的映射
        :param idx2word: 索引到单词的映射
        :param max_length: 最大句子长度，超出则截断
        :param pad_idx: 填充 token 索引
        :param bos_idx: 句子起始 token 索引
        :param eos_idx: 句子结束 token 索引
        :param unk_idx: 未知单词索引
        """
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_length = max_length
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.unk_idx = unk_idx

    def encode(self, text_list, padding_first=False, add_bos=True, add_eos=True, return_mask=False):
        """
        将文本列表编码为索引列表。
        :param text_list: 文本列表，每个元素是一个单词列表
        :param padding_first: 是否将 [PAD] 填充到句首
        :param add_bos: 是否添加 [BOS] 起始符号
        :param add_eos: 是否添加 [EOS] 结束符号
        :param return_mask: 是否返回 mask
        :return: 编码后的 input_ids 或 (input_ids, masks)
        """
        max_length = min(self.max_length, add_eos + add_bos + max([len(text) for text in text_list]))
        indices_list = []

        for text in text_list:
            indices = [self.word2idx.get(word, self.unk_idx) for word in text[:max_length - add_bos - add_eos]]
            if add_bos: indices = [self.bos_idx] + indices
            if add_eos: indices = indices + [self.eos_idx]

            # 填充到 max_length
            if padding_first:
                indices = [self.pad_idx] * (max_length - len(indices)) + indices
            else:
                indices = indices + [self.pad_idx] * (max_length - len(indices))

            indices_list.append(indices)

        input_ids = torch.tensor(indices_list) # 转换为 tensor
        masks = (input_ids == self.pad_idx).to(dtype=torch.int64)  # 生成 mask，标记 padding 部分

        return input_ids if not return_mask else (input_ids, masks)

    def decode(self, indices_list, remove_bos=True, remove_eos=True, remove_pad=True, split=False):
        """
        解码索引列表为文本列表。
        :param indices_list: 索引列表
        :param remove_bos: 是否移除 [BOS]
        :param remove_eos: 是否移除 [EOS]
        :param remove_pad: 是否移除 [PAD]
        :param split: 是否返回分词列表
        :return: 解码后的文本列表
        """
        text_list = []

        for indices in indices_list:
            text = []
            for index in indices:
                word = self.idx2word.get(index, "[UNK]")  # 获取单词
                if remove_bos and word == "[BOS]": continue
                if remove_eos and word == "[EOS]": break
                if remove_pad and word == "[PAD]": break
                text.append(word)

            text_list.append(" ".join(text) if not split else text)

        return text_list
#%%
# 创建英文和中文的tokenizer
en_tokenizer = Tokenizer(word2idx=en_word2idx, idx2word=en_idx2word)
zh_tokenizer = Tokenizer(word2idx=zh_word2idx, idx2word=zh_idx2word)

# 输出词表大小
en_vocab_size = len(en_word2idx)
zh_vocab_size = len(zh_word2idx)
# print("en_vocab_size: {}".format(en_vocab_size))  # 打印英文词表大小
# print("zh_vocab_size: {}".format(zh_vocab_size))  # 打印中文词表大小
#%% md
# ### Transformer Batch Sampler
#%%
class SampleInfo:
    def __init__(self, i, lens):
        """
        记录文本对的序号和长度信息。

        :param i: 文本对的序号。
        :param lens: 文本对的长度，包含源语言和目标语言的长度。
        """
        self.i = i
        # 加一是为了考虑填充的特殊词元，lens[0] 和 lens[1] 分别表示源语言和目标语言的长度
        self.max_len = max(lens[0], lens[1]) + 1
        self.src_len = lens[0] + 1
        self.trg_len = lens[1] + 1


class TokenBatchCreator:
    def __init__(self, batch_size):
        """
        根据词元数目限制批量大小，并初始化批量存储结构。

        :param batch_size: 批量的最大大小。
        """
        self._batch = []  # 当前处理的样本
        self.max_len = -1  # 当前批量的最大长度
        self._batch_size = batch_size  # 批量大小限制

    def append(self, info: SampleInfo):
        """
        将样本信息添加到批量中。如果当前批量大小超过限制，则返回当前批量并创建新批量。

        :param info: 包含文本对长度信息的 SampleInfo 对象。
        :return: 当前批量样本，如果超过限制，则返回并重置批量，否则返回 None。
        """
        cur_len = info.max_len  # 当前样本的最大长度
        max_len = max(self.max_len, cur_len)  # 更新当前批量的最大长度

        # 如果当前批量加入新样本后超过限制，返回当前批量并重置
        if max_len * (len(self._batch) + 1) > self._batch_size:
            self._batch, result = [], self._batch  # 保存当前样本并清空
            self._batch.append(info)  # 新批量的第一条样本
            self.max_len = cur_len  # 当前批量的最大长度为新样本的最大长度
            return result
        else:
            self.max_len = max_len
            self._batch.append(info)  # 将新样本加入当前批量
            return None

    @property
    def batch(self):
        return self._batch
#%%
from torch.utils.data import BatchSampler
import numpy as np

class TransformerBatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 shuffle_batch=False,
                 clip_last_batch=False,
                 seed=0):
        """
        批量采样器，用于按批次生成样本。

        :param dataset: 数据集
        :param batch_size: 每个批次的样本数量
        :param shuffle_batch: 是否对批次进行洗牌
        :param clip_last_batch: 是否裁剪最后一个批次的数据（如果样本数不足一个完整批次）
        :param seed: 随机数种子，用于保证可重复性
        """
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle_batch = shuffle_batch
        self._clip_last_batch = clip_last_batch
        self._seed = seed
        self._random = np.random
        self._random.seed(seed)

        self._sample_infos = []
        # 创建样本信息列表，包含每个样本的索引和长度信息
        for i, data in enumerate(self._dataset):
            lens = [len(data[0]), len(data[1])]  # 计算样本的源语言和目标语言的长度
            self._sample_infos.append(SampleInfo(i, lens)) # 保存为 [索引，样本长度]的格式

    def __iter__(self):
        """
        对数据集中的样本进行排序，并使用 TokenBatchCreator 生成批次。

        排序规则：先按源语言长度排序，若源语言长度相同，再按目标语言长度排序。
        生成的批次如果未裁剪最后一批，则将剩余样本组成最后一个批次。
        如果需要洗牌，则对批次进行洗牌。

        :yield: 每个批次的样本在数据集中的索引
        """
        # 按源语言和目标语言长度排序
        infos = sorted(self._sample_infos, key=lambda x: (x.src_len, x.trg_len))
        batch_infos = []
        batch_creator = TokenBatchCreator(self._batch_size)  # 批量生成器

        # 逐个样本加入批量生成器
        for info in infos:
            batch = batch_creator.append(info)
            if batch is not None:
                batch_infos.append(batch)  # 每当批次满足要求时，保存当前批次

        # 如果未裁剪最后一个批次且有剩余样本，则将剩余样本作为最后一个批次
        if not self._clip_last_batch and len(batch_creator.batch) != 0:
            batch_infos.append(batch_creator.batch)

        # 打乱批次顺序
        if self._shuffle_batch:
            self._random.shuffle(batch_infos)

        # 记录批次数量
        self.batch_number = len(batch_infos)

        # 生成批次中的样本索引
        for batch in batch_infos:
            batch_indices = [info.i for info in batch]  # 获取当前批次的样本索引
            yield batch_indices

    def __len__(self):
        """
        返回批次数量
        """
        if hasattr(self, "batch_number"):
            return self.batch_number

        # 如果没有记录批次数量，计算批次样本数量
        batch_number = (len(self._dataset) + self._batch_size - 1) // self._batch_size
        return batch_number
#%% md
# ### DataLoader
#%%
from functools import partial # 固定collate_fct的tokenizer参数
def collate_fct(batch, en_tokenizer, zh_tokenizer):
    # 分别对源语言(英文)和目标语言(中文)进行处理
    src_words = [pair[0].split() for pair in batch]
    trg_words = [pair[1].split() for pair in batch]

    # 使用英文tokenizer处理源语言
    encoder_inputs, encoder_inputs_mask = en_tokenizer.encode(
        src_words, padding_first=False, add_bos=True, add_eos=True, return_mask=True
    )

    # 使用中文tokenizer处理目标语言
    decoder_inputs = zh_tokenizer.encode(
        trg_words, padding_first=False, add_bos=True, add_eos=False, return_mask=False,
    )

    decoder_labels, decoder_labels_mask = zh_tokenizer.encode(
        trg_words, padding_first=False, add_bos=False, add_eos=True, return_mask=True
    )

    return {
        "encoder_inputs": encoder_inputs.to(device=device),
        "encoder_inputs_mask": encoder_inputs_mask.to(device=device),
        "decoder_inputs": decoder_inputs.to(device=device),
        "decoder_labels": decoder_labels.to(device=device),
        "decoder_labels_mask": decoder_labels_mask.to(device=device),
    }
#%% md
# ## 定义模型
#%% md
# ### Embedding
#%%
class TransformerEmbedding(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        # 获取配置中的超参数
        self.vocab_size = vocab_size
        self.hidden_size = config["d_model"]
        self.pad_idx = config["pad_idx"]
        dropout_rate = config["dropout"]
        self.max_length = config["max_length"]

        # 词嵌入层，padding_idx为pad的索引
        self.word_embedding = nn.Embedding(self.vocab_size, self.hidden_size, padding_idx=self.pad_idx)
        # 位置嵌入层，权重由get_positional_encoding计算得到
        self.pos_embedding = nn.Embedding(
            self.max_length,
            self.hidden_size,
            _weight=self.get_positional_encoding(self.max_length, self.hidden_size)
        )
        self.pos_embedding.weight.requires_grad_(False)  # 位置编码不可训练
        self.dropout = nn.Dropout(dropout_rate)  # Dropout层

    def get_word_embedding_weights(self):
        # 返回词嵌入层的权重
        return self.word_embedding.weight

    @classmethod
    def get_positional_encoding(self, max_length, hidden_size):
        """
        计算位置编码
        使用正弦和余弦函数生成位置编码矩阵
        """
        pe = torch.zeros(max_length, hidden_size)
        position = torch.arange(0, max_length).unsqueeze(1)  # 位置索引
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * -(torch.log(torch.Tensor([10000.0])) / hidden_size)
        )
        # 填充位置编码矩阵
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列为sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列为cos
        return pe

    def forward(self, input_ids):
        """
        前向传播：词向量与位置编码加和
        """
        seq_len = input_ids.shape[1]  # 序列长度
        assert seq_len <= self.max_length, f"序列长度超出最大限制 {self.max_length}"

        # 生成位置id
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device) # [seq_len]
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # [batch_size, seq_len]

        # 获取词嵌入和位置编码
        word_embeds = self.word_embedding(input_ids) # [batch_size, seq_len, hidden_size]
        pos_embeds = self.pos_embedding(position_ids) # [batch_size, seq_len, hidden_size]
        embeds = word_embeds + pos_embeds  # 加和词向量和位置编码
        embeds = self.dropout(embeds)  # 应用dropout
        return embeds  # [batch_size, seq_len, hidden_size]


def plot_position_embedding(position_embedding):
    """
    绘制位置编码矩阵
    """
    plt.pcolormesh(position_embedding)  # 绘制矩阵
    plt.xlabel('Depth')  # x轴为深度
    plt.ylabel('Position')  # y轴为位置
    plt.colorbar()  # 显示颜色条
    plt.show()  # 显示图像
#%% md
# ### Transformer
#%% md
# #### Scaled Dot Product Attention
#%%
from dataclasses import dataclass
from typing import Optional, Tuple

Tensor = torch.Tensor

@dataclass
class AttentionOutput:
    hidden_states: Tensor  # 注意力层的最终输出
    attn_scores: Tensor    # 计算得到的注意力权重

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 获取配置中的超参数
        self.hidden_size = config["d_model"]
        self.num_heads = config["num_heads"]

        # 确保 hidden_size 可以被 num_heads 整除
        assert self.hidden_size % self.num_heads == 0, "Hidden size must be divisible by num_heads"

        self.head_dim = self.hidden_size // self.num_heads  # 每个头的维度

        # 定义线性变换层
        self.Wq = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.Wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        """
        将输入张量拆分为多个头
        """
        bs, seq_len, _ = x.shape
        x = x.view(bs, seq_len, self.num_heads, self.head_dim)  # 调整形状
        return x.permute(0, 2, 1, 3)  # 调整维度顺序

    def _merge_heads(self, x: Tensor) -> Tensor:
        """
        合并多个头的输出
        """
        bs, _, seq_len, _ = x.shape
        return x.permute(0, 2, 1, 3).reshape(bs, seq_len, self.hidden_size)

    def forward(self, querys, keys, values, attn_mask=None) -> AttentionOutput:
        """
        前向传播计算注意力机制
        """
        # 线性变换并拆分为多个头
        querys = self._split_heads(self.Wq(querys))  # [batch_size, num_heads, seq_len, head_dim]
        keys = self._split_heads(self.Wk(keys))      # [batch_size, num_heads, seq_len, head_dim]
        values = self._split_heads(self.Wv(values))  # [batch_size, num_heads, seq_len, head_dim]

        # 计算 Q 和 K 之间的点积注意力分数
        qk_logits = torch.matmul(querys, keys.mT)  # [batch_size, num_heads, seq_len, seq_len]

        # 如果提供了mask，应用它
        if attn_mask is not None:
            attn_mask = attn_mask[:, :, : querys.shape[-2], : keys.shape[-2]]  # 调整mask的大小
            qk_logits += attn_mask * -1e9  # mask部分置为负无穷

        # 计算注意力权重
        attn_scores = F.softmax(qk_logits / (self.head_dim**0.5), dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 加权求和
        embeds = torch.matmul(attn_scores, values)  # [batch_size, num_heads, seq_len, head_dim]

        # 合并头的输出并通过输出投影层
        embeds = self.Wo(self._merge_heads(embeds))  # [batch_size, seq_len, hidden_size]

        return AttentionOutput(hidden_states=embeds, attn_scores=attn_scores)
#%% md
# #### Transformer Block
#%%
@dataclass
class TransformerBlockOutput:
    # 当前块的输出（隐藏状态）和自/交叉注意力的分数
    hidden_states: Tensor
    self_attn_scores: Tensor
    cross_attn_scores: Optional[Tensor] = None

class TransformerBlock(nn.Module):
    def __init__(self, config, add_cross_attention=False):
        super().__init__()

        # 获取配置中的超参数
        self.hidden_size = config["d_model"]
        self.num_heads = config["num_heads"]
        dropout_rate = config["dropout"]
        ffn_dim = config["dim_feedforward"]
        eps = config["layer_norm_eps"]

        # 自注意力机制
        self.self_atten = MultiHeadAttention(config)
        self.self_ln = nn.LayerNorm(self.hidden_size, eps=eps)
        self.self_dropout = nn.Dropout(dropout_rate)

        # 交叉注意力机制（仅解码器中使用）
        if add_cross_attention:
            self.cross_atten = MultiHeadAttention(config)
            self.cross_ln = nn.LayerNorm(self.hidden_size, eps=eps)
            self.cross_dropout = nn.Dropout(dropout_rate)
        else:
            self.cross_atten = None

        # 前馈神经网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, self.hidden_size),
        )
        self.ffn_ln = nn.LayerNorm(self.hidden_size, eps=eps)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, attn_mask=None, encoder_outputs=None, cross_attn_mask=None):
        # 自注意力机制
        self_atten_output = self.self_atten(
            hidden_states, hidden_states, hidden_states, attn_mask
        )
        self_embeds = self.self_ln(
            hidden_states + self.self_dropout(self_atten_output.hidden_states)
        )

        # 交叉注意力机制（解码器中）
        if self.cross_atten is not None:
            assert encoder_outputs is not None  # 使用交叉注意力时，必须传入编码器输出
            cross_atten_output = self.cross_atten(
                self_embeds, encoder_outputs, encoder_outputs, cross_attn_mask
            )
            cross_embeds = self.cross_ln(
                self_embeds + self.cross_dropout(cross_atten_output.hidden_states)
            )

        # 前馈神经网络（FFN）
        embeds = cross_embeds if self.cross_atten is not None else self_embeds
        ffn_output = self.ffn(embeds)
        embeds = self.ffn_ln(embeds + self.ffn_dropout(ffn_output))

        # 返回 TransformerBlockOutput
        return TransformerBlockOutput(
            hidden_states=embeds,
            self_attn_scores=self_atten_output.attn_scores,
            cross_attn_scores=cross_atten_output.attn_scores if self.cross_atten is not None else None,
        )
#%% md
# #### Encoder
#%%
from typing import List

@dataclass  # 存储TransformerEncoder的输出
class TransformerEncoderOutput:
    last_hidden_states: Tensor  # 最后一层的隐藏状态，包含上下文信息
    attn_scores: List[Tensor]   # 每层的注意力分数，用于分析每层的关注点

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 获取编码器层数
        self.num_layers = config["num_encoder_layers"]

        # 创建包含多个TransformerBlock的列表，每个TransformerBlock代表编码器的一层
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(self.num_layers)]
        )

    def forward(self, encoder_inputs_embeds, attn_mask=None) -> TransformerEncoderOutput:
        attn_scores = []  # 存储每层的自注意力分数
        embeds = encoder_inputs_embeds  # 输入嵌入作为编码器的第一层输入

        # 遍历每一层TransformerBlock
        for layer in self.layers:
            block_outputs = layer(embeds, attn_mask=attn_mask)  # 当前层输出

            embeds = block_outputs.hidden_states  # 更新下一层输入
            attn_scores.append(block_outputs.self_attn_scores)  # 保存注意力分数

        # 返回最后一层输出和所有层的注意力分数
        return TransformerEncoderOutput(
            last_hidden_states=embeds,  # 最后一层的输出
            attn_scores=attn_scores  # 所有层的注意力分数
        )
#%% md
# #### Decoder
#%%
@dataclass
class TransformerDecoderOutput:
    last_hidden_states: Tensor  # 最后一层的隐藏状态
    self_attn_scores: List[Tensor]  # 每层的自注意力分数
    cross_attn_scores: List[Tensor]  # 每层的交叉注意力分数

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 获取解码器层数
        self.num_layers = config["num_decoder_layers"]

        # 创建多个TransformerBlock，每层都有交叉注意力机制
        self.layers = nn.ModuleList(
            [TransformerBlock(config, add_cross_attention=True) for _ in range(self.num_layers)]
        )

    def forward(self, decoder_inputs_embeds, encoder_outputs, attn_mask=None, cross_attn_mask=None) -> TransformerDecoderOutput:
        self_attn_scores = []  # 存储每层的自注意力分数
        cross_attn_scores = []  # 存储每层的交叉注意力分数
        embeds = decoder_inputs_embeds  # 解码器输入嵌入

        # 遍历每一层的TransformerBlock
        for layer in self.layers:
            # 前向传播，通过自注意力和交叉注意力机制
            block_outputs = layer(
                embeds,
                attn_mask=attn_mask,
                encoder_outputs=encoder_outputs,
                cross_attn_mask=cross_attn_mask,
            )
            embeds = block_outputs.hidden_states  # 更新输入为当前层输出

            self_attn_scores.append(block_outputs.self_attn_scores)  # 保存自注意力分数
            cross_attn_scores.append(block_outputs.cross_attn_scores)  # 保存交叉注意力分数

        # 返回最后一层的隐藏状态和每层的注意力分数
        return TransformerDecoderOutput(
            last_hidden_states=embeds,
            self_attn_scores=self_attn_scores,
            cross_attn_scores=cross_attn_scores
        )
#%% md
# #### Transformer Model
#%%
@dataclass
class TransformerOutput:
    logits: Tensor  # 模型的预测输出 logits
    encoder_last_hidden_states: Tensor  # 编码器的最终隐藏状态
    encoder_attn_scores: List[Tensor]  # 编码器各层的自注意力得分
    decoder_last_hidden_states: Tensor  # 解码器的最终隐藏状态
    decoder_self_attn_scores: List[Tensor]  # 解码器自注意力得分
    decoder_cross_attn_scores: List[Tensor]  # 解码器交叉注意力得分
    preds: Optional[Tensor] = None  # 推理时的预测结果

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 模型的各项配置初始化
        self.hidden_size = config["d_model"]  # Transformer模型中的隐藏层维度
        self.num_encoder_layers = config["num_encoder_layers"]  # 编码器层数
        self.num_decoder_layers = config["num_decoder_layers"]  # 解码器层数
        self.pad_idx = config["pad_idx"]  # padding 标记的索引
        self.bos_idx = config["bos_idx"]  # 句子开始标记的索引
        self.eos_idx = config["eos_idx"]  # 句子结束标记的索引
        self.en_vocab_size = config["en_vocab_size"]  # 英文词表大小
        self.zh_vocab_size = config["zh_vocab_size"]  # 中文词表大小
        self.dropout_rate = config["dropout"]  # Dropout比例
        self.max_length = config["max_length"]  # 最大序列长度

        # 初始化源语言(英文)和目标语言(中文)的嵌入层
        self.src_embedding = TransformerEmbedding(config, vocab_size=self.en_vocab_size)
        self.trg_embedding = TransformerEmbedding(config, vocab_size=self.zh_vocab_size)

        # 输出层的线性变换，输出的维度为中文词表大小
        self.linear = nn.Linear(self.hidden_size, self.zh_vocab_size)

        # 初始化编码器和解码器
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重"""
        for p in self.parameters():
            if p.dim() > 1:
                # 使用 Xavier 均匀分布初始化权重
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """生成解码器的下三角掩码，用于自回归解码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 0).transpose(-1, -2).bool()  # 下三角掩码
        return mask

    def forward(self, encoder_inputs, decoder_inputs, encoder_inputs_mask=None) -> TransformerOutput:
        """
        Transformer前向传播
        1. 将输入的源语言和目标语言进行嵌入。
        2. 通过编码器处理源语言输入。
        3. 通过解码器生成目标语言的输出。
        """
        if encoder_inputs_mask is None:
            # 如果没有提供mask，则根据padding索引生成
            encoder_inputs_mask = encoder_inputs.eq(self.pad_idx)
        encoder_inputs_mask = encoder_inputs_mask.unsqueeze(1).unsqueeze(2)  # 扩展mask维度，以适应多头注意力

        # 生成解码器掩码（防止信息泄漏）
        look_ahead_mask = self.generate_square_subsequent_mask(decoder_inputs.shape[1]).unsqueeze(0).unsqueeze(0).to(decoder_inputs.device)
        decoder_inputs_mask = decoder_inputs.eq(self.pad_idx).unsqueeze(1).unsqueeze(2) + look_ahead_mask

        # 编码阶段：将源语言输入映射到嵌入空间
        encoder_inputs_embeds = self.src_embedding(encoder_inputs)
        encoder_outputs = self.encoder(encoder_inputs_embeds, encoder_inputs_mask)

        # 解码阶段：将目标语言输入映射到嵌入空间
        decoder_inputs_embeds = self.trg_embedding(decoder_inputs)
        decoder_outputs = self.decoder(
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=encoder_outputs.last_hidden_states,
            attn_mask=decoder_inputs_mask,
            cross_attn_mask=encoder_inputs_mask,
        )

        # 将解码器的输出通过线性变换映射到目标语言的词表大小
        logits = self.linear(decoder_outputs.last_hidden_states)

        return TransformerOutput(
            logits=logits,
            encoder_last_hidden_states=encoder_outputs.last_hidden_states,
            encoder_attn_scores=encoder_outputs.attn_scores,
            decoder_last_hidden_states=decoder_outputs.last_hidden_states,
            decoder_self_attn_scores=decoder_outputs.self_attn_scores,
            decoder_cross_attn_scores=decoder_outputs.cross_attn_scores,
        )

    @torch.no_grad()
    def infer(self, encoder_inputs, encoder_inputs_mask=None) -> Tensor:
        """推理：生成目标语言的翻译结果"""
        if encoder_inputs_mask is None:
            encoder_inputs_mask = encoder_inputs.eq(self.pad_idx)  # 根据padding生成mask
        encoder_inputs_mask = encoder_inputs_mask.unsqueeze(1).unsqueeze(2)

        # 生成解码器的掩码（自回归）
        look_ahead_mask = self.generate_square_subsequent_mask(self.max_length).unsqueeze(0).unsqueeze(0).to(encoder_inputs.device)

        # 编码阶段
        encoder_inputs_embeds = self.src_embedding(encoder_inputs)
        encoder_outputs = self.encoder(encoder_inputs_embeds)

        # 解码阶段：生成目标语言翻译
        decoder_inputs = torch.Tensor([self.bos_idx] * encoder_inputs.shape[0]).reshape(-1, 1).long().to(device=encoder_inputs.device)
        for cur_len in tqdm(range(1, self.max_length + 1)):
            decoder_inputs_embeds = self.trg_embedding(decoder_inputs)
            decoder_outputs = self.decoder(
                decoder_inputs_embeds=decoder_inputs_embeds,
                encoder_outputs=encoder_outputs.last_hidden_states,
                attn_mask=look_ahead_mask[:, :, :cur_len, :cur_len],
            )
            logits = self.linear(decoder_outputs.last_hidden_states)
            next_token = logits.argmax(dim=-1)[:, -1:]  # 选择下一个最可能的token
            decoder_inputs = torch.cat([decoder_inputs, next_token], dim=-1)  # 将token加入解码输入中

            # 如果所有样本的解码器输出达到结束标记，则停止解码
            if all((decoder_inputs == self.eos_idx).sum(dim=-1) > 0):
                break

        return TransformerOutput(
            preds=decoder_inputs[:, 1:],  # 排除开始标记，返回最终的预测结果
            logits=logits,
            encoder_last_hidden_states=encoder_outputs.last_hidden_states,
            encoder_attn_scores=encoder_outputs.attn_scores,
            decoder_last_hidden_states=decoder_outputs.last_hidden_states,
            decoder_self_attn_scores=decoder_outputs.self_attn_scores,
            decoder_cross_attn_scores=decoder_outputs.cross_attn_scores,
        )