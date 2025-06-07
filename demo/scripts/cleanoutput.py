import pandas as pd

# Load your CSV
df = pd.read_csv("VSPW_segmentation_metrics_mobilenetv3_0_classes.csv")

# Exclude the first column (e.g., Video Name) from numeric checking
non_numeric_cols = ["Video Name"]

# Find columns where all values (from row 2 onward) are 0
columns_to_drop = []
for col in df.columns:
    if col in non_numeric_cols:
        continue
    if (df[col].iloc[1:] == 0).all():
        columns_to_drop.append(col)

# Drop the identified columns
df_cleaned = df.drop(columns=columns_to_drop)

# Save the cleaned CSV
df_cleaned.to_csv("VSPW_segmentation_metrics_mobilenetv3_0_classes_filtered.csv", index=False)
