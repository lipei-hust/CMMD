import pandas as pd
from utils import agg
import os
from pathlib import Path
import re
from multiprocessing import Process, Queue


def data_preprocess():
    output_dir = r".\preprocessed_data\original"
    filepath = Path(output_dir)
    filepath.mkdir(parents=True, exist_ok=True)
    for root, dir, files in os.walk(r"D:\squeeze dataset\encrypted_F\B_cuboid_layer_1_n_ele_1"):
        for filename in files:
            if not re.search(r"\.(a|b)\.csv", filename):
                continue
            print(f"process {filename}")
            path = os.path.join(root, filename)
            data = pd.read_csv(path)
            for s in ["a", "b", "c", "d"]:
                data[s].replace(regex=True, inplace=True, to_replace=r"[a-d]", value=r"")
            data.to_csv(
                os.path.join(output_dir, filename),
            )

    print("data preprocess done")


def derived_data(data1, data2):
    new_data = pd.DataFrame(columns=["real", "predict", "a", "b", "c", "d"])
    for k in ["a", "b", "c", "d"]:
        new_data[k] = data1[k]
    new_data["real"] = data1["real"] / data2["real"]
    new_data["predict"] = data1["predict"] / data2["predict"]
    new_data.fillna("0", inplace=True)
    return new_data
def agg_1(data: pd.DataFrame):
    dimension_dict = {
        "a": data["a"],
        "b": data["b"],
        "c": data["c"],
        "d": data["d"],
    }
    result_data = pd.DataFrame()
    for k in dimension_dict:
        # for each of ['a','b','c','d']
        # AGG each
        items = ["a", "b", "c", "d"]
        items.remove(k)
        v0 = items[0]
        v1 = items[1]
        v2 = items[2]
        # v0 is dimension, such as "a"
        # vv0 is value of this dimension, sucn as "a5"
        for vv0 in dimension_dict[v0].unique():
            for vv1 in dimension_dict[v1].unique():
                for vv2 in dimension_dict[v2].unique():
                    mask = (data[v0] == vv0) & (data[v1] == vv1) & (data[v2] == vv2)
                    selected_data = data[mask]
                    new_data = agg(selected_data, [k])
                    result_data = result_data.append(new_data, ignore_index=True)

    return result_data
def agg_3(data: pd.DataFrame):
    dimension_dict = {
        "a": data["a"],
        "b": data["b"],
        "c": data["c"],
        "d": data["d"],
    }
    result_data = pd.DataFrame()
    for i in range(4):
        items = ["a", "b", "c", "d"]
        v = items[i]
        items.remove(v)
        for vv in dimension_dict[v].unique():
            mask = data[v] == vv
            selected_data = data[mask]
            new_data = agg(selected_data, items)
            result_data = result_data.append(new_data, ignore_index=True)
    return result_data
data_dir = r".\preprocessed_data"


def agg_4(data: pd.DataFrame):
    return agg(data, ["a", "b", "c", "d"])

def agg_1_file(q: Queue):
    while True:
        filename = q.get()
        print(f"agg {filename}...")

        i = os.path.join(data_dir, "original")
        o = os.path.join(data_dir, "agg1")

        data = pd.read_csv(os.path.join(i, filename))
        r = agg_1(data)
        r.to_csv(os.path.join(o, filename))

        print(f"agg {filename} done")


def agg_3_file(q: Queue):
    while True:
        filename = q.get()
        print(f"agg {filename}...")

        i = os.path.join(data_dir, "original")
        o = os.path.join(data_dir, "agg3")

        data = pd.read_csv(os.path.join(i, filename))
        r = agg_3(data)
        r.to_csv(os.path.join(o, filename))

        print(f"agg {filename} done")


def agg_4_file(q: Queue):
    while True:
        filename = q.get()
        print(f"agg {filename}...")

        i = os.path.join(data_dir, "original")
        o = os.path.join(data_dir, "agg4")

        data = pd.read_csv(os.path.join(i, filename))
        r = agg_4(data)
        r.to_csv(os.path.join(o, filename))

        print(f"agg {filename} done")


def from_timestamp_to_node(q, data_by_timestamp: list[pd.DataFrame], data_type, data_agg_type):
    while True:
        node_index: int = q.get()
        node_data = pd.DataFrame()
        for data in data_by_timestamp:
            node_data = node_data.append(data.iloc[node_index], ignore_index=True)
        node_index_str = str(node_index + 1).zfill(5)
        path = os.path.join(
            data_dir, f"{data_agg_type}_by_node", f"{node_index_str}.{data_type}.csv"
        )
        node_data.to_csv(path)
        if (node_index + 1) % 100 == 0:
            print(f"node {node_index + 1} finished")


if __name__ == "__main__":
    # compute derived data
    # input_dir = os.path.join(data_dir, "agg4")

    # filenums = []
    # for root, dir, files in os.walk(input_dir):
    #     for filename in files:
    #         searchObject = re.search(r"(\d*)\.(a|b)\.csv", filename)
    #         if not searchObject:
    #             continue
    #         num = searchObject.group(1)
    #         if num in filenums:
    #             continue
    #         filenums.append(num)
    #         data1 = pd.read_csv(os.path.join(input_dir, f"{num}.a.csv"))
    #         data2 = pd.read_csv(os.path.join(input_dir, f"{num}.b.csv"))
    #         r = derived_data(data1, data2)
    #         r.to_csv(os.path.join(input_dir, f"{num}.derived.csv"))

    # agg 1
    # q = Queue(maxsize=1)

    # threads = [Process(target=agg_1_file, args=(q,)) for i in range(16)]
    # for thread in threads:
    #     thread.start()

    # input_dir = os.path.join(data_dir, "original")
    # for root, dir, files in os.walk(input_dir):
    #     for filename in files:
    #         searchObject = re.search(r"(\d*)\.(a|b)\.csv", filename)
    #         if not searchObject:
    #             continue
    #         q.put(filename)

    # q.join()

    # agg 3
    # q = Queue(maxsize=1)

    # threads = [Process(target=agg_3_file, args=(q,)) for i in range(16)]
    # for thread in threads:
    #     thread.start()

    # input_dir = os.path.join(data_dir, "original")
    # for root, dir, files in os.walk(input_dir):
    #     for filename in files:
    #         searchObject = re.search(r"(\d*)\.(a|b)\.csv", filename)
    #         if not searchObject:
    #             continue
    #         q.put(filename)

    # agg 4
    # q = Queue(maxsize=1)

    # threads = [Process(target=agg_4_file, args=(q,)) for i in range(16)]
    # for thread in threads:
    #     thread.start()

    # input_dir = os.path.join(data_dir, "original")
    # for root, dir, files in os.walk(input_dir):
    #     for filename in files:
    #         searchObject = re.search(r"(\d*)\.(a|b)\.csv", filename)
    #         if not searchObject:
    #             continue
    #         q.put(filename)

    # change data group from timestamp to node
    data_agg_type = "original"
    data_type = "derived"
    data_by_timestamp: list[pd.DataFrame] = []
    for root, dir, files in os.walk(os.path.join(data_dir, data_agg_type)):
        for filename in files:
            searchObject = re.search(f"(\d*)\.{data_type}\.csv", filename)
            if not searchObject:
                continue
            data = pd.read_csv(os.path.join(data_dir, data_agg_type, filename))
            data = data.drop("Unnamed: 0", axis=1)
            data_by_timestamp.append(data)

    q = Queue(maxsize=1)

    processes = [
        Process(
            target=from_timestamp_to_node, args=(q, data_by_timestamp, data_type, data_agg_type)
        )
        for _ in range(15)
    ]
    for process in processes:
        process.start()

    for i in range(21600):
        q.put(i)
    q.join()
