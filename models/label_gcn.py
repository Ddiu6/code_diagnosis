"""标签共现图卷积模块 (Label Co-occurrence GCN, ML-GCN 风格)。

Reference: Chen et al., "Multi-Label Image Recognition with Graph Convolutional Networks", CVPR 2019.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def build_label_adjacency(train_records, num_classes: int, threshold: float = 0.4) -> np.ndarray:
    """
    统计训练集标签共现条件概率矩阵 P(病j | 病i)，构建邻接矩阵。

    眼底疾病有明确的临床共现规律（玻璃体后脱离↔玻璃体混浊、
    玻璃体积血↔视网膜脱离、后巩膜葡萄肿↔玻璃体后脱离），该矩阵把这些
    先验显式编码，供 GCN 建模标签间依赖。

    Args:
        train_records: list of dict，每个含 "label" 字段（[C] multi-hot）
        num_classes:   类别数 C
        threshold:     弱共现边阈值（低于此值置 0，去除噪声边）

    Returns:
        [C, C] 邻接矩阵（非负，行 i 表示"有病 i 时，病 j 的共现条件概率"）
    """
    cooccur = np.zeros((num_classes, num_classes), dtype=np.float32)
    label_count = np.zeros(num_classes, dtype=np.float32)

    for r in train_records:
        label = np.asarray(r["label"], dtype=np.float32)
        for i in range(num_classes):
            if label[i] == 1:
                label_count[i] += 1
                for j in range(num_classes):
                    if label[j] == 1:
                        cooccur[i][j] += 1

    adj = np.zeros((num_classes, num_classes), dtype=np.float32)
    for i in range(num_classes):
        if label_count[i] > 0:
            adj[i] = cooccur[i] / label_count[i]

    # 阈值化去噪：去除弱共现边
    adj[adj < threshold] = 0.0
    return adj


class LabelCooccurrenceGCN(nn.Module):
    """
    标签共现图卷积模块 (ML-GCN, CVPR 2019)。

    用训练集标签共现统计构建邻接矩阵，通过 2 层 GCN 建模疾病标签间的
    临床共现关系，得到"标签增强表示"，替代普通分类器的权重矩阵，
    让分类 logits 融合标签间先验（尤其帮助长尾类借力共现的大类）。

    流程:
        label_emb [C, D]（可学习标签嵌入）
          → 2 层 GCN（邻接矩阵 A，归一化+自环）→ 标签增强表示 [C, D]
        pooled [B, D] @ 标签表示^T + bias → logits [B, C]

    Args:
        num_classes: 类别数 C
        embed_dim:   特征维度 D
        adj:         [C, C] 标签共现邻接矩阵（numpy 或 torch）
        dropout:     dropout
    """

    def __init__(self, num_classes: int, embed_dim: int, adj, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # 归一化邻接矩阵（加自环 + 对称归一化），register_buffer 不参与梯度
        self.register_buffer("adj", self._normalize_adj(adj))

        # 可学习标签嵌入（标准初始化，量级与 GCN 层 bias 匹配，避免被 bias 淹没导致类间无区分）
        self.label_emb = nn.Parameter(torch.empty(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.label_emb)

        # 2 层 GCN
        self.gcn1 = nn.Linear(embed_dim, embed_dim)
        self.gcn2 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def _normalize_adj(self, adj) -> torch.Tensor:
        if isinstance(adj, np.ndarray):
            A = torch.from_numpy(adj.astype(np.float32))
        else:
            A = adj.clone().float()
        # 加自环
        A = A + torch.eye(A.size(0), device=A.device)
        # 对称归一化 D^-1/2 A D^-1/2
        D = A.sum(dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        return D_inv_sqrt.view(-1, 1) * A * D_inv_sqrt.view(1, -1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: [B, D] 图像池化特征
        Returns:
            logits: [B, C]
        """
        h = self.label_emb                                   # [C, D]
        h = self.relu(self.gcn1(self.adj @ h))               # [C, D]
        h = self.dropout(h)
        h = self.gcn2(self.adj @ h)                          # [C, D]
        logits = pooled @ h.t() + self.bias                  # [B, C]
        return logits
