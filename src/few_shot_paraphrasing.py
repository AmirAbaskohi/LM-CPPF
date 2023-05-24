import pandas as pd


def gpt(text, task_name):
    df = pd.read_csv(f"../paraphrased_data/paraphrased_gpt3_{task_name}.csv", index=False)
    return df[df["text"==text]]["paraphrased"]

def opt(text, task_name):
    df = pd.read_csv(f"../paraphrased_data/paraphrased_opt175b_{task_name}.csv", index=False)
    return df[df["text"==text]]["paraphrased"]