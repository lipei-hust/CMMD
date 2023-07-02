import torch
import numpy as np
import os
import pandas as pd
import re
import re
import pickle

device = "cuda"

X: torch.Tensor = torch.ones(1, 100, 2).to(device)
Y: torch.Tensor = torch.ones(1, 100, 1).to(device)
data_dir = r".\preprocessed_data"
original_dir = os.path.join(data_dir, "original_by_node")
agg4_dir = os.path.join(data_dir, "agg4_by_node")
agg4_data1 = np.array(pd.read_csv(os.path.join(agg4_dir, "1.a.csv"))["real"])
agg4_data2 = np.array(pd.read_csv(os.path.join(agg4_dir, "1.b.csv"))["real"])
agg4_data_derived = pd.read_csv(os.path.join(agg4_dir, "1.derived.csv"))["real"]
X[0] = torch.tensor(np.append(agg4_data1, agg4_data2)).view(2, 100).t()
Y[0] = torch.tensor(np.array(agg4_data_derived)).view(100, 1)
for root, dir, files in os.walk(original_dir):
    for filename in files:
        searchObject = re.search(r"(\d*)\.derived\.csv", filename)
        if not searchObject:
            continue
        node_index = int(searchObject.group(1))

        node_index_str = str(node_index).zfill(5)
        data1 = np.array(pd.read_csv(os.path.join(original_dir, f"{node_index_str}.a.csv"))["real"])
        data2 = np.array(pd.read_csv(os.path.join(original_dir, f"{node_index_str}.b.csv"))["real"])
        new_x = torch.tensor(np.append(data1, data2)).view(2, 100).t().view(1, 100, 2)
        X = torch.vstack((X, new_x))
        if node_index % 100 == 0:
            print(f"X {node_index_str} completed")

for root, dir, files in os.walk(original_dir):
    for filename in files:
        searchObject = re.search(r"(\d*)\.derived\.csv", filename)
        if not searchObject:
            continue
        node_index = int(searchObject.group(1))

        node_index_str = str(node_index).zfill(5)
        data = np.array(
            pd.read_csv(os.path.join(original_dir, f"{node_index_str}.derived.csv"))["real"]
        )
        new_y = torch.tensor(np.array(data)).view(1, 100, 1)
        Y = torch.vstack((Y, new_y))
        if node_index % 100 == 0:
            print(f"Y {node_index_str} completed")

print("dumping X...")
with open("tensorX.pkl", "wb") as f:
    pickle.dump(X, f)

print("dumping Y...")
with open("tensorY.pkl", "wb") as f:
    pickle.dump(Y, f)
