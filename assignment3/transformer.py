
# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)  # 先创建在CPU
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)  # 先创建在CPU
        
    # 添加设备移动方法
    def to(self, device):
        self.input_tensor = self.input_tensor.to(device)
        self.output_tensor = self.output_tensor.to(device)
        return self


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal,num_heads,dropout,num_classes, num_layers,device):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer：词嵌入和Transformer层的维度
        :param d_internal: see TransformerLayer：Transformer层内部前馈网络维度
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()


        #我们由于任务是分类任务，所以不需要解码器
        #需要解码器的任务：序列生成类任务（如机器翻译、文本生成），需要逐步生成输出序列
        #此处我们只需要借助编码器提取特征，然后套一个分类头即可
        #输入序列 → [词嵌入] → [位置编码] → [N层Transformer编码器] → [分类头]

        self.device = device if device is not None else torch.device("cpu")
        self.d_model=d_model
        self.embedding=nn.Embedding(vocab_size, d_model).to(self.device)
        self.pos_encoder = PositionalEncoding(d_model, num_positions).to(self.device)  # 确保位置编码器也在GPU上

        #nn.ModuleList需要手动写forward,nn.Sequential会自动调用forward
        #我们此处使用ModuleList而不是Sequential，因为我们需要收集每层的注意力图
        self.transformer_layers = nn.ModuleList([TransformerLayer(d_model, d_internal,num_heads,dropout) for _ in range(num_layers)]).to(self.device)

        # 分类头：将d_model维向量映射到num_classes维
        self.classifier = nn.Linear(d_model, num_classes).to(self.device)


        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """


        #indices：输入句子的 单词索引序列（即经过分词和词表映射后的数字ID）。
        #示例： 假设词表为 {"我":0, "爱":1, "你":2, "<unk>":3}，那么句子 "我 爱 你" 会转换为：
        #indices = torch.LongTensor([0, 1, 2])  
        #在模型中的处理流程：通过词嵌入层：将索引转换为向量。也就是变为3个向量（每个向量维度为 d_model）

        #确保都在GPU上
        if indices.device != next(self.parameters()).device:
                indices = indices.to(next(self.parameters()).device)

        # 添加batch维度处理
        if indices.dim() == 1:
            indices = indices.unsqueeze(0)  # [seq_len] -> [1, seq_len]
            
        # 1. 词嵌入
        X = self.embedding(indices)  # [batch_size, seq_len, d_model]
        
        # 2. 位置编码
        X = self.pos_encoder(X)  # [batch_size, seq_len, d_model]
        
        # 3. 通过多层Transformer编码器并收集注意力图
        attn_maps = []
        for layer in self.transformer_layers:
            X, attn = layer(X)  # x shape: [seq_len, d_model]
            attn_maps.append(attn)  # attn shape: [seq_len, seq_len, num_heads]
        
        # 4. 分类头
        logits = self.classifier(X)  # [seq_len, num_classes]
        

        if indices.dim() == 2 and indices.size(0) == 1:  # 如果是单样本
            return F.log_softmax(logits, dim=-1).squeeze(0), [a.squeeze(0) for a in attn_maps]
        else:
            return F.log_softmax(logits, dim=-1), attn_maps
        
        


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal,num_heads,dropout):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.head_dim=self.d_internal//self.num_heads

        # 定义可学习参数（无偏置）
        self.W_Q = nn.Parameter(torch.Tensor(d_model, d_internal))
        self.W_K = nn.Parameter(torch.Tensor(d_model, d_internal))
        self.W_V = nn.Parameter(torch.Tensor(d_model, d_internal))

        # 初始化权重（例如 Xavier）
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)

        # 1. 注意力机制后的Dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # 2. FFN层内的Dropout
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Dropout(dropout),  # FFN第一层后
            nn.Linear(d_internal, d_model),
            nn.Dropout(dropout)   # FFN输出层后
        )
        

        self.proj_out = nn.Linear(d_internal, d_model)  # 专门用于处理维度改变的


        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # raise Exception("Implement me")

    def forward(self, input_vecs):
        """
        :param input_vecs: an input tensor of shape [seq len, d_model]
        :return: a tuple of two elements:
            - a tensor of shape [seq len, d_model] representing the log probabilities of each position in the input
            - a tensor of shape [seq len, seq len], representing the attention map for this layer
        """

        # 添加batch处理逻辑（保持原有变量名X）
        if input_vecs.dim() == 2:
            # 无batch情况 [seq_len, d_model]
            X = input_vecs.unsqueeze(0)  # 转为[1, seq_len, d_model]
            need_squeeze = True
        else:
            # 有batch情况 [batch_size, seq_len, d_model]
            X = input_vecs
            need_squeeze = False

        #得到Q，K，V
        Q=X@self.W_Q
        K=X@self.W_K
        V=X@self.W_V


        # 进行分头，实现多头注意力:[seq_len, d_model] -> [seq_len, num_heads, head_dim]
        # 输入序列：seq_len = 3（例如 3 个字符 ["A", "B", "C"]）
        # 模型维度：d_model = 8（每个字符的向量维度是 8）
        # 头数：num_heads = 2（分成 2 个头）
        # 每个头的维度：head_dim = d_model / num_heads = 4
        
        # 示例输入：（3个位置，每个位置8维向量）
        # Q = torch.tensor([
        #     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # "A" 的 Query
        #     [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],  # "B" 的 Query
        #     [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]   # "C" 的 Query
        # ])
        
        # 分头后的 Q:
        # tensor([
        #     [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],  # "A" 的两个头
        #     [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8]],  # "B" 的两个头
        #     [[2.1, 2.2, 2.3, 2.4], [2.5, 2.6, 2.7, 2.8]]   # "C" 的两个头
        # ])

        # 进行分头（修改为支持batch的reshape方式）
        batch_size, seq_len, _ = Q.shape
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)



        #计算注意力分数:
        #因为是多头，所以我们需要按照头来分块计算：
        # 对 第一个头（head=0），提取 Q 和 K：
        # Q_head0 = Q[:, 0, :]  # shape [3, 4]
        # K_head0 = K[:, 0, :]  # shape [3, 4]
        # 然后计算W0 = Q_head0 * K_head0^T，同样的，计算W1 = Q_head1 * K_head1^T，以此类推

        # Q 的第一个头（h=0）
        # Q_head0 = torch.tensor([
        #     [0.1, 0.2, 0.3, 0.4],  # "A" 的 Query
        #     [1.1, 1.2, 1.3, 1.4],  # "B" 的 Query
        #     [2.1, 2.2, 2.3, 2.4]   # "C" 的 Query
        # ])

        # # K 的第一个头（h=0）
        # K_head0 = torch.tensor([
        #     [0.9, 0.8, 0.7, 0.6],  # "A" 的 Key
        #     [1.9, 1.8, 1.7, 1.6],  # "B" 的 Key
        #     [2.9, 2.8, 2.7, 2.6]   # "C" 的 Key
        # ])

        # 手动计算 在第一个注意力头的情况下，"A" 对 "B" 的分数
        # score_A_B = sum(Q_head0[0] * K_head0[1])  (0.1*1.9 + 0.2*1.8 + 0.3*1.7 + 0.4*1.6) = 1.7
        # 因此最后合为矩阵就是：
        # 在第一个注意力头的情况下:
        # [
        #     [0.90, 1.70, 2.50],  # "A" 对 "A", "B", "C"
        #     [3.70, 7.30, 10.90],  # "B" 对 "A", "B", "C"
        #     [6.50, 12.90, 19.30]  # "C" 对 "A", "B", "C"
        # ]
        #然后我们可以去算在第二个注意力头的情况下，各个对应的分数
        #这样子第一个头弄出来的是(3,3)的矩阵，第二个头弄出来的也是(3,3)的矩阵
        #然后我们把他们堆叠到一起即可，最后得到(3,3,2)的矩阵


        # 自动遍历所有头（h）
        # 对每个头计算 Q[:,h,:] @ K[:,h,:].T
        # 将结果按 h 维度堆叠
        attn_scores = torch.einsum("bqhd,bkhd->bqkh", Q, K)  #高端操作！！！！ # [batch_size,seq_len, seq_len, num_heads]


        # 缩放
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=1)  # [seq_len, seq_len, num_heads]

        #然后是对于每个头，我们去得到Y：
        output_of_attention = torch.einsum("bqkh,bkhd->bqhd", attn_weights, V)

        # 合并多头:将多头的输出拼接回 [seq_len, d_internal]：
        # 合并多头（修改为支持batch的reshape方式）
        output_of_attention = output_of_attention.transpose(1, 2).reshape(batch_size, seq_len, self.d_internal)
        output_of_attention = self.proj_out(output_of_attention)  # [batch, seq_len, d_model]
        output_of_attention = self.attn_dropout(output_of_attention)  # [batch, seq_len, d_model]


        res_of_attention = output_of_attention + X  # 残差连接1
        res_of_attention = self.norm1(res_of_attention)   # LayerNorm1



        output_of_ffn=self.ffn(res_of_attention)  

        res_of_ffn=output_of_ffn + res_of_attention  # 残差连接2
        res_of_ffn=self.norm2(res_of_ffn)   # LayerNorm2


        # 修改后代码（合并多头注意力）：
        if need_squeeze:
            # 合并多头注意力（取平均）
            combined_attn = attn_weights.mean(dim=-1).squeeze(0)  # [seq_len, seq_len]
            return res_of_ffn.squeeze(0), combined_attn
        else:
            # 合并多头注意力（取平均）
            combined_attn = attn_weights.mean(dim=1)  # [batch, seq_len, seq_len]
            return res_of_ffn, combined_attn


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=True):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """

        # 此处我们是固定了句子长度为20，因此我们此处的位置编码的长度也是固定为20，更多情况下我们是取最大长度的句子作为位置编码的长度
        # 输入矩阵：[seq_len=2, d_model=2]（2个位置，每个位置是一个2维向量）
        # 位置编码矩阵：[num_positions=3, d_model=2]（预计算好的3个位置编码，每一个位置编码的维度需要与输入的向量维度匹配）
        # x = torch.tensor([[1,2], [3,4]])       # [2,2]
        # pe = torch.tensor([[0.1,0.1], [0.2,0.2], [0.3,0.3]])  # [3,2]
        # output = x + pe[:2]  # 取前2个位置编码
        # 最后结果：[[1.1,2.1], [3.2,4.2]]

        #当然，上面的是非批处理模式，就是一个句子那么就是对应一个矩阵：rows=20（因为一个句子我们固定有20个单词），cols=每个单词对应的向量维度
        #如果是批处理模式，那么就是每个矩阵每个矩阵的那样处理，本质上是一样的

        super().__init__()
        # Dict size

        #创建一个位置编码的矩阵，然后这个矩阵里面的值是可以学习的，动态编码位置
        #行数：num_positions=20（处理的最大序列长度）
        #列数：d_model=（每个位置的编码维度）
        self.emb = nn.Embedding(num_positions, d_model)

        self.batched = batched

    def forward(self, x):
        # 获取输入序列长度
        seq_len = x.size(-2) if self.batched else x.size(0)
        
        # 创建位置索引（确保与输入x在同一设备上）
        indices = torch.arange(seq_len, dtype=torch.long, device=x.device)
        
        # 获取位置编码
        pos_emb = self.emb(indices)
        
        # 根据是否批处理调整维度
        if self.batched:
            return x + pos_emb.unsqueeze(0)  # 添加batch维度
        else:
            return x + pos_emb





class LetterCountingDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
# class LetterCountingDataset(Dataset):
#     def __init__(self, examples, augment_prob=0.5):
#         """
#         Args:
#             examples: List[LetterCountingExample] 从train_bundles传入
#             augment_prob: 数据增强概率
#         """
#         self.examples = examples
#         self.augment_prob = augment_prob
#         self.vocab = [chr(ord('a') + i) for i in range(26)] + [' ']
        
#     def __len__(self):
#         return len(self.examples)
    
#     def __getitem__(self, idx):
#         ex = self.examples[idx]  # 这是原始的LetterCountingExample对象
        
#         if random.random() < self.augment_prob:
#             # 创建增强版本（不修改原始example）
#             chars = list(ex.input)
#             # 随机替换1-3个非空格字符
#             for _ in range(random.randint(1, 3)):
#                 idx = random.choice([i for i, c in enumerate(chars) if c != ' '])
#                 chars[idx] = random.choice(self.vocab[:-1])
            
#             # 创建新的output
#             new_output = np.zeros(len(chars))
#             counts = {}
#             for i, c in enumerate(chars):
#                 new_output[i] = min(2, counts.get(c, 0))
#                 counts[c] = counts.get(c, 0) + 1
                
#             # 返回增强后的数据（保持张量格式）
#             input_tensor = torch.LongTensor([ex.vocab_index.index_of(c) for c in chars])
#             output_tensor = torch.LongTensor(new_output)
#             return input_tensor, output_tensor
            
#         # 不增强时返回原始数据
#         return ex.input_tensor, ex.output_tensor


def collate_fn(batch):
    inputs = torch.stack([ex.input_tensor for ex in batch])
    targets = torch.stack([ex.output_tensor for ex in batch])
    return inputs, targets


# def collate_fn(batch):
#     # batch现在是元组列表(input_tensor, output_tensor)
#     inputs = torch.stack([item[0] for item in batch])
#     targets = torch.stack([item[1] for item in batch])
#     return inputs, targets


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Transformer(
        vocab_size=getattr(args, 'vocab_size', 27),
        num_positions=getattr(args, 'num_positions', 20),
        d_model=getattr(args, 'd_model', 128),
        d_internal=getattr(args, 'd_internal', 512),
        num_heads=getattr(args, 'num_heads', 4),
        dropout=getattr(args, 'dropout', 0.2),
        num_classes=getattr(args, 'num_classes', 3),
        num_layers=getattr(args, 'num_layers', 3),
        device=device
    )
    model.to(device)

    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01
    )

    # 学习率调度
    # 建议添加warmup（前4000步线性增加学习率）
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(
            (step+1)* (1e-4 / 10000),  # warmup阶段
            (step+1)**-0.5            # 后续衰减
        )
    )

    # 损失函数（带类别权重）
    class_counts = torch.bincount(torch.cat([ex.output_tensor for ex in train]))
    class_weights = 1. / (class_counts.float() + 1e-6)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=0.1
    )

    # 创建DataLoader
    train_dataset = LetterCountingDataset(train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 训练循环
    best_val_acc = 0
    for epoch in range(getattr(args, 'num_epochs', 100)):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=True)
        
        for batch_inputs, batch_targets in progress_bar:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            log_probs, _ = model(batch_inputs)
            
            # 计算损失
            loss = criterion(
                log_probs.view(-1, getattr(args, 'num_classes', 3)),
                batch_targets.view(-1)
            )
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # 统计信息
            epoch_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(log_probs, dim=-1)
                correct += (preds == batch_targets).sum().item()
                total += batch_targets.numel()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{epoch_loss/(len(progress_bar)+1):.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        # 验证阶段
        val_acc = evaluate(model, dev, device)
        print(f"Epoch {epoch+1} | "
              f"Train Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Train Acc: {100*correct/total:.2f}% | "
              f"Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    # 训练完成后，准备解码
    model = model.cpu()
    for ex in dev:
        ex.input_tensor = ex.input_tensor.cpu()

    decode(model, dev, do_print=True, do_plot_attn=False)

    return model


def evaluate(model, dev_data, device, batch_size=32):
    model.eval()
    correct = 0
    total = 0
    
    # 创建验证集DataLoader
    dev_dataset = LetterCountingDataset(dev_data)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    with torch.no_grad():
        for batch_inputs, batch_targets in dev_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            log_probs, _ = model(batch_inputs)
            preds = torch.argmax(log_probs, dim=-1)
            
            correct += (preds == batch_targets).sum().item()
            total += batch_targets.numel()
    
    return 100 * correct / total


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
