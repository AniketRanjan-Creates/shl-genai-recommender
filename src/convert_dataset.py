import pandas as pd

# Read Excel file
df = pd.read_excel("data/Gen_AI Dataset.xlsx")

# Keep only the two required columns
df = df[["Query", "Assessment_url"]]

# Remove any empty rows or duplicates
df = df.dropna().drop_duplicates()

# Save as CSV
df.to_csv("data/labeled_train.csv", index=False)

print("✅ Done! Saved as data/labeled_train.csv")
