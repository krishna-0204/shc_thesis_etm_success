import pandas as pd

old = pd.read_csv("data/processed/clean_features_old.csv", nrows=0).columns.tolist()
new = pd.read_csv("data/processed/clean_features.csv", nrows=0).columns.tolist()

print("Exact order & names match:", old == new)
print("Only-in-new:", set(new) - set(old))
print("Only-in-old:", set(old) - set(new))
print("First index where they differ:", next((i for i,(a,b) in enumerate(zip(old,new)) if a!=b), None))
