import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# Load interaction and pen assignment data
interactions = pd.read_csv("interactions.csv")
pen_assignments = pd.read_csv("pen_assignments.csv")

# Add week column if missing
if 'week' not in interactions.columns:
    SECONDS_PER_WEEK = 604800
    interactions['week'] = (interactions['start_ts'] // SECONDS_PER_WEEK).astype(int)

# Merge pen_id for cow_i
pen_map = pen_assignments[['week', 'cow_id', 'pen_id']]
interactions = interactions.merge(
    pen_map.rename(columns={'cow_id': 'cow_i', 'pen_id': 'pen_id'}),
    on=['week', 'cow_i'], how='left'
)
# Merge pen_id for cow_j
interactions = interactions.merge(
    pen_map.rename(columns={'cow_id': 'cow_j', 'pen_id': 'pen_j'}),
    on=['week', 'cow_j'], how='left'
)
# Keep only interactions where both cows are in the same pen
interactions = interactions[interactions['pen_id'] == interactions['pen_j']]

# Ensure output directory exists
os.makedirs("graphs", exist_ok=True)

# Plot and save social network graphs per pen per week
for week in sorted(interactions['week'].unique()):
    for pen_id in sorted(interactions['pen_id'].unique()):
        week_pen_df = interactions[(interactions['week'] == week) & (interactions['pen_id'] == pen_id)]
        if week_pen_df.empty:
            continue
        G = nx.Graph()
        # Count number of interactions per pair
        interaction_counts = week_pen_df.groupby(['cow_i', 'cow_j']).size().reset_index(name='count')
        for _, row in interaction_counts.iterrows():
            cow1, cow2, count = row['cow_i'], row['cow_j'], row['count']
            G.add_edge(cow1, cow2, count=count)
        plt.figure(figsize=(8, 6))
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', edgecolors='black', linewidths=1)
        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
        edge_labels = {(u, v): f"{G[u][v]['count']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        plt.title(f"Social Network: Pen {pen_id}, Week {week}")
        plt.axis('off')
        fname = f"social_network_pen{pen_id}_week{week}.png"
        plt.savefig(f"graphs/{fname}", bbox_inches='tight')
        plt.close()