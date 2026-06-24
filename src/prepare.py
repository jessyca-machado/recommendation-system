import pandas as pd
import os
os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv("data/raw/events.csv")

df['datetime'] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

cutoff = df["datetime"].quantile(0.8)

train = df[df["datetime"] < cutoff]
test = df[df["datetime"] >= cutoff]

train.to_csv("data/processed/train.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)
