import torch
import torch.nn as nn
import pickle
import random
import numpy as np

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

from GAT_Class import GAT

input_feature_size = 2
output_feature_size = 1
embedding_dimension = 8
lr = 5e-4
epochs = 1000
device = "cuda"
batch_size = 1

X: torch.Tensor = ""
Y: torch.Tensor = ""
with open("tensorX.pkl", "rb") as f:
    X = pickle.load(f).to(torch.float32).to(device)
with open("tensorY.pkl", "rb") as f:
    Y = pickle.load(f).to(torch.float32).to(device)
def data_iter():
    sample_index = 0
    while sample_index < X.shape[1]:
        x = X[:, sample_index : sample_index + batch_size]
        y = Y[:, sample_index : sample_index + batch_size]
        sample_index += batch_size
        yield x.view([21601, 2]), y.view([21601, 1])
model = GAT(
    in_feats=input_feature_size, hidden_feats=embedding_dimension, out_feats=output_feature_size
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()
record_loss = []
model.train()
for epoch in range(epochs):
    for x, y in data_iter():
        # 使用所有节点(全图)进行前向传播计算
        logits = model(x)
        # 计算损失值
        l = loss(input=logits, target=y)
        # 进行反向传播计算
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print(f"epoch: {epoch} loss: {l.item()}")
    record_loss.append(l.item())

torch.save(model, f"./models/lr={lr} batch={batch_size} seed={seed} 4layers.model")
title = f"lr={lr} batch={batch_size} seed={seed} 4layers"
with open(f"./result/{title}.pkl", "wb") as f:
    pickle.dump(record_loss, f)
