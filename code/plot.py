import matplotlib.pyplot as plt
import pickle

record = []
lr = 5e-4
batch = 1
seed = 10
title = f"lr={lr} batch={batch} seed={seed}"
with open(f"./result/{title}.pkl", "rb") as f:
    record = pickle.load(f)

x = range(1000)
fig = plt.figure()
main_f = fig.add_axes([0.15, 0.1, 0.8, 0.8])
main_f.plot(x, record)
main_f.set_title(title)
main_f.set_xlabel("epochs")
main_f.set_ylabel("loss")

sub_f = fig.add_axes([0.45, 0.3, 0.4, 0.25])
# sub_f.set_ylim(0, 1)
sub_f.plot(x[499:], record[499:])

plt.savefig(fname=f"./result/{title}.png")
