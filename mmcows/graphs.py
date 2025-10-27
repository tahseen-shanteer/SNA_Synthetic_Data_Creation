import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# --------------------------
# Load interaction data
# --------------------------
interactions = pd.read_csv("interactions.csv")

# --------------------------
# Define week as 7 days from the first timestamp
# --------------------------
SECONDS_PER_WEEK = 7 * 24 * 60 * 60
min_ts = interactions['start_ts'].min()
interactions['week'] = ((interactions['start_ts'] - min_ts) // SECONDS_PER_WEEK).astype(int) + 1  # week 1, week 2, ...

# --------------------------
# Assign all cows to the same pen
# --------------------------
interactions['pen_id'] = 1  # single pen assumption

# --------------------------
# Ensure output directory exists
# --------------------------
os.makedirs("graphs", exist_ok=True)

# --------------------------
# Plot and save social network graphs per week
# --------------------------
for week in sorted(interactions['week'].unique()):
    week_df = interactions[interactions['week'] == week]
    if week_df.empty:
        continue

    G = nx.Graph()

    # Count number of interactions per pair
    interaction_counts = week_df.groupby(['cow_i', 'cow_j']).size().reset_index(name='count')

    for _, row in interaction_counts.iterrows():
        cow1, cow2, count = row['cow_i'], row['cow_j'], row['count']
        G.add_edge(cow1, cow2, count=count)

    # Calculate edge weights inversely proportional to interaction count (more interactions = shorter edge)
    max_count = interaction_counts['count'].max()
    min_length = 0.2
    max_length = 2.0
    edge_lengths = {}
    for _, row in interaction_counts.iterrows():
        count = row['count']
        # Inverse scaling: more interactions = shorter edge
        length = max_length - (count - 1) / (max_count - 1 + 1e-9) * (max_length - min_length)
        edge_lengths[(row['cow_i'], row['cow_j'])] = length

    # Set the 'distance' attribute for each edge
    for (u, v), length in edge_lengths.items():
        G[u][v]['distance'] = length
    pos = nx.spring_layout(G, k=None, weight='distance', iterations=100, scale=2.0, seed=42)

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', edgecolors='black', linewidths=1)
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    plt.title(f"Social Network: Pen 1, Week {week}")
    plt.axis('off')
    fname = f"social_network_week{week}.png"
    plt.savefig(f"graphs/{fname}", bbox_inches='tight')
    plt.close()