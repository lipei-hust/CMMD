import pandas as pd


def agg(selected_data: pd.DataFrame, dimensions: list[str]):
    real = selected_data["real"].sum()
    predict = selected_data["predict"].sum()
    new_data = pd.DataFrame(
        {
            "real": [real],
            "predict": [predict],
        },
    )
    strs = ["a", "b", "c", "d"]
    for d in dimensions:
        new_data[d] = 0
        strs.remove(d)

    for s in strs:
        new_data[s] = selected_data[s].iloc[0]

    return new_data
