import pandas as pd
g = pd.read_csv("data/labeled_train.csv")
qs = sorted(g["Query"].dropna().unique().tolist())
for i, q in enumerate(qs, 1):
    print(f"{i:02d}. {q[:120].replace(chr(10),' ')}")
