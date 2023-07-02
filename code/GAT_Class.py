from torch import nn
import torch
import dgl
import dgl.nn as dglnn

heads = 8
# define graph
edge_begin = []
edge_end = []
leaf_node_num = 21600
device = "cuda"
for i in range(leaf_node_num):
    edge_begin.append(i + 1)
    edge_end.append(0)
    edge_begin.append(i + 1)
    edge_end.append(i + 1)

# edge_begin.append(0)
# edge_end.append(0)
edge_begin = torch.tensor(edge_begin)
edge_end = torch.tensor(edge_end)
graph = dgl.graph((edge_begin, edge_end)).to(device)

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.GATConv(
            in_feats=in_feats,
            out_feats=hidden_feats,
            num_heads=heads,
            # allow_zero_in_degree=True,
        )
        self.conv2 = dglnn.GATConv(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            num_heads=heads,
            # allow_zero_in_degree=True,
        )
        self.conv3 = dglnn.GATConv(
            in_feats=hidden_feats,
            out_feats=hidden_feats,
            num_heads=heads,
            # allow_zero_in_degree=True,
        )
        self.leakyrelu = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Linear(hidden_feats, out_feats)
    def forward(self, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = torch.sum(h, 1)
        h = self.leakyrelu(h)
        h = self.conv2(graph, h)
        h = torch.sum(h, 1)
        h = self.leakyrelu(h)
        h = self.conv3(graph, h)
        h = torch.sum(h, 1)
        h = self.leakyrelu(h)
        # h = torch.sum(h, 1)
        h = self.mlp(h)
        return h
