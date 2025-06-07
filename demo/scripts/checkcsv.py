import pandas as pd

# Load CSV
df = pd.read_csv("results.csv")

# Get car-related columns
car_columns = [col for col in df.columns if "car" in col.lower()]
car_scores = df[["Video Name", "Fusion Threshold"] + car_columns]

# Show result
print(car_scores)

# Optional: save to CSV
car_scores.to_csv("car_scores.csv", index=False)
