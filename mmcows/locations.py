import os
import glob
import pandas as pd

# Path to the UWB folder
uwb_folder = "uwb"

# List to hold DataFrames
dfs = []

# Loop through each cow folder (e.g., T01, T02, ...)
for cow_folder in glob.glob(os.path.join(uwb_folder, "T*")):
    cow_id = os.path.basename(cow_folder)
    # Loop through each CSV file in the cow folder
    for file in glob.glob(os.path.join(cow_folder, "*.csv")):
        df = pd.read_csv(file)
        df["cow_id"] = cow_id
        df["source_file"] = os.path.basename(file)
        dfs.append(df)

# Concatenate all DataFrames
all_locations = pd.concat(dfs, ignore_index=True)

# Save to a single CSV file
all_locations.to_csv("all_cow_locations.csv", index=False)
print("Merged all cow location files into all_cow_locations.csv")