import torch
from torch.nn import Module
import pandas as pd
import numpy as np
from GAT_Class import GAT

device = "cuda"
lr = 5e-4
batch_size = 1
seed = 10

timeslice = 1450653900
f_root = pd.read_csv(f"./preprocessed_data/agg4/{timeslice}.derived.csv")["predict"].sum()
v_root = pd.read_csv(f"./preprocessed_data/agg4/{timeslice}.derived.csv")["real"].sum()

v_root_a = pd.read_csv(f"./preprocessed_data/agg4/{timeslice}.a.csv")["real"].sum()
v_root_b = pd.read_csv(f"./preprocessed_data/agg4/{timeslice}.b.csv")["real"].sum()

v_leaf_a = pd.read_csv(f"./preprocessed_data/original/{timeslice}.a.csv")["real"]
v_leaf_b = pd.read_csv(f"./preprocessed_data/original/{timeslice}.b.csv")["real"]

model: Module = torch.load(f"./models/lr={lr} batch={batch_size} seed={seed}.model").to(device)
model.eval()

x = torch.tensor(np.append(v_leaf_a, v_leaf_b)).view((2, 21600)).t()
x0 = torch.tensor([v_root_a, v_root_b]).view((1, 2))
x = torch.cat((x0, x)).to(torch.float32).to(device)


fy = model(x)
print(fy[0].item())
print(v_root)
