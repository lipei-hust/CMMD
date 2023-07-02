from sko.GA import GA, crossover
import pandas as pd
import numpy as np
import torch
from torch.nn import Module
from GAT_Class import GAT

tc = 0.5
tf = 0.1
beta = 1
device = "cuda"

lr = 5e-4
batch_size = 1
seed = 10

timeslice = 1450653900
f_root = pd.read_csv(f"./preprocessed_data/agg4/{timeslice}.derived.csv")["predict"].sum()
v_root = pd.read_csv(f"./preprocessed_data/agg4/{timeslice}.derived.csv")["real"].sum()
f_leaf_a = pd.read_csv(f"./preprocessed_data/original/{timeslice}.a.csv")["predict"]
v_leaf_a = pd.read_csv(f"./preprocessed_data/original/{timeslice}.a.csv")["real"]
f_leaf_b = pd.read_csv(f"./preprocessed_data/original/{timeslice}.b.csv")["predict"]
v_leaf_b = pd.read_csv(f"./preprocessed_data/original/{timeslice}.b.csv")["real"]
model: Module = torch.load(f"./models/lr={lr} batch={batch_size} seed={seed}.model").to(device)
model.eval()
def evaluate(p: list):
    new_v_a = v_leaf_a.copy()
    new_v_b = v_leaf_b.copy()
    sum = 0
    for i in range(21600):
        if p[i] == 1:
            sum += 1
            new_v_a[i] = f_leaf_a[i]
            new_v_b[i] = f_leaf_b[i]
    input_data = torch.tensor(np.append(new_v_a, new_v_b)).view((2, 21600)).t()
    input_data = torch.cat((torch.zeros(1, 2), input_data)).to(torch.float32).to(device)
    out_data = model(input_data)
    out_data = abs(out_data[0].item() - f_root) / abs(v_root - f_root) + beta * sum
    return out_data


ga = GA(func=evaluate, n_dim=21600, prob_mut=tf, lb=0, ub=1, max_iter=10, size_pop=50, precision=1)
ga.register(operator_name="crossover", operator=crossover.crossover_2point_prob, crossover_prob=tc)
best_x, best_y = ga.run()
print(len(best_x))
sum = 0
for v in best_x:
    if v == 1:
        sum += 1
print(sum)
print(best_y)
