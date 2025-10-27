import pandas as pd
import numpy as np
from itertools import combinations

def load_cow_data(all_file):
    """
    Load all cow location data from a single CSV.
    Returns dict {cow_id: DataFrame}
    """
    df = pd.read_csv(all_file)

    # Keep only needed columns
    df = df[["timestamp", "coord_x_cm", "coord_y_cm", "coord_z_cm", "cow_id"]]

    # Split into dict of DataFrames per cow
    cow_data = {
        cow_id: sub_df.drop(columns=["cow_id"]).reset_index(drop=True)
        for cow_id, sub_df in df.groupby("cow_id")
    }
    return cow_data

def find_interactions(cow_data, d=150, t=60):
    """Find interactions given threshold distance d (cm) and min duration t (seconds)."""
    results = []
    cow_ids = list(cow_data.keys())

    for cow_i, cow_j in combinations(cow_ids, 2):
        df_i = cow_data[cow_i]
        df_j = cow_data[cow_j]

        # Merge on timestamp (assumes synchronized samples, e.g. every 15s)
        merged = pd.merge(df_i, df_j, on="timestamp", suffixes=(f"_{cow_i}", f"_{cow_j}"))

        # Compute distances
        merged["distance"] = np.sqrt(
            (merged[f"coord_x_cm_{cow_i}"] - merged[f"coord_x_cm_{cow_j}"])**2 +
            (merged[f"coord_y_cm_{cow_i}"] - merged[f"coord_y_cm_{cow_j}"])**2 +
            (merged[f"coord_z_cm_{cow_i}"] - merged[f"coord_z_cm_{cow_j}"])**2
        )

        # Detect continuous intervals
        in_interaction = False
        start_ts, start_loc_i, start_loc_j = None, None, None

        for idx, row in merged.iterrows():
            if row["distance"] <= d:
                if not in_interaction:
                    # start interaction
                    in_interaction = True
                    start_ts = row["timestamp"]
                    start_loc_i = (
                        row[f"coord_x_cm_{cow_i}"],
                        row[f"coord_y_cm_{cow_i}"],
                        row[f"coord_z_cm_{cow_i}"]
                    )
                    start_loc_j = (
                        row[f"coord_x_cm_{cow_j}"],
                        row[f"coord_y_cm_{cow_j}"],
                        row[f"coord_z_cm_{cow_j}"]
                    )
            else:
                if in_interaction:
                    # end interaction
                    end_ts = merged.loc[idx-1, "timestamp"]
                    duration = end_ts - start_ts
                    if duration >= t:
                        results.append([
                            cow_i, cow_j, start_ts, end_ts,
                            *start_loc_i, *start_loc_j
                        ])
                    in_interaction = False

        # If still interacting at the end
        if in_interaction:
            end_ts = merged.iloc[-1]["timestamp"]
            duration = end_ts - start_ts
            if duration >= t:
                results.append([
                    cow_i, cow_j, start_ts, end_ts,
                    *start_loc_i, *start_loc_j
                ])

    return pd.DataFrame(results, columns=[
        "cow_i", "cow_j", "start_ts", "end_ts",
        "initial_loc_cow_i_x_cm", "initial_loc_cow_i_y_cm", "initial_loc_cow_i_z_cm",
        "initial_loc_cow_j_x_cm", "initial_loc_cow_j_y_cm", "initial_loc_cow_j_z_cm"
    ])

def generate_interactions_csv(all_file="all_cow_locations.csv", output_file="interactions.csv", d=250, t=60):
    cow_data = load_cow_data(all_file)
    interactions = find_interactions(cow_data, d, t)
    interactions.to_csv(output_file, index=False)
    print(f"Saved interactions to {output_file}")

# Example usage:
generate_interactions_csv("all_cow_locations.csv", "interactions.csv", d=250, t=60)
