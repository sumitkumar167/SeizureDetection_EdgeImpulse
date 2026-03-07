import pandas as pd

df = pd.read_csv("seizure_features.csv")

seizure = df[df["label"] == 1]
normal = df[df["label"] == 0]

# sample equal number of normal windows 1:1 ratio
normal_sample = normal.sample(len(seizure), random_state=42)

balanced = pd.concat([seizure, normal_sample])
balanced = balanced.sample(frac=1)

balanced.to_csv("balanced_seizure_features.csv", index=False)