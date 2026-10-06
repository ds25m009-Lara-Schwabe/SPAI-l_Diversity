import pandas as pd

df = pd.read_csv("data/cahousing/cahousing.csv", sep=";")

print(df["ocean_proximity"].value_counts())
print()
print("Unique values:", df["ocean_proximity"].nunique())