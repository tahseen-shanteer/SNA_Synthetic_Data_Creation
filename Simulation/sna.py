import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

# --------------------------
# Load Data
# --------------------------
interactions = pd.read_csv("interactions.csv")
cow_registry = pd.read_csv("cow_registry.csv")
pen_assignments = pd.read_csv("pen_assignments.csv")

# Ensure timestamps are integers (seconds from 0)
interactions['start_ts'] = interactions['start_ts'].astype(int)
interactions['end_ts']   = interactions['end_ts'].astype(int)

# Compute week number (0-based)
SECONDS_PER_WEEK = 604800
interactions['week'] = (interactions['start_ts'] // SECONDS_PER_WEEK)

# Merge with pen assignments (for cow_i and cow_j separately)
pen_map = pen_assignments[['week','cow_id','pen_id']]

interactions = interactions.merge(
    pen_map.rename(columns={'cow_id':'cow_i','pen_id':'pen_i'}),
    on=['week','cow_i'], how='left'
)
interactions = interactions.merge(
    pen_map.rename(columns={'cow_id':'cow_j','pen_id':'pen_j'}),
    on=['week','cow_j'], how='left'
)

# Keep only interactions where both cows are in the SAME pen
interactions = interactions[interactions['pen_i'] == interactions['pen_j']]
interactions['pen_id'] = interactions['pen_i']

# --------------------------
# Build Graphs Per Pen Per Week
# --------------------------
def build_graphs(interactions, week_num):
    graphs = {}
    week_df = interactions[interactions['week'] == week_num]
    
    for pen_id, pen_df in week_df.groupby('pen_id'):
        G = nx.Graph()
        for _, row in pen_df.iterrows():
            cow1, cow2 = row['cow_i'], row['cow_j']
            duration = row['end_ts'] - row['start_ts']  # interaction weight
            if G.has_edge(cow1, cow2):
                G[cow1][cow2]['weight'] += duration
            else:
                G.add_edge(cow1, cow2, weight=duration, zone=row['zone'])
        graphs[pen_id] = G
    return graphs

# --------------------------
# Compute Centrality Metrics
# --------------------------
def compute_metrics(G):
    degree = dict(G.degree())                         # unweighted degree
    strength = dict(G.degree(weight='weight'))        # weighted degree
    betweenness = nx.betweenness_centrality(G, weight='weight')
    if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G, weight='weight')
        except Exception:
            eigenvector = {node: 0.0 for node in G.nodes()}
    else:
        eigenvector = {node: 0.0 for node in G.nodes()}

    metrics = pd.DataFrame({
        'degree': pd.Series(degree),
        'strength': pd.Series(strength),
        'betweenness': pd.Series(betweenness),
        'eigenvector': pd.Series(eigenvector)
    })
    return metrics

# --------------------------
# Compute Combined Risk Metric
# --------------------------
def compute_combined_risk(all_metrics_df):
    """
    Computes combined isolation and conflict risk metrics for each cow, pen, and week:
    s' = log(1 + strength/3600)
    combined_risk_isolation = 0.4 * minmax(degree) + 0.6 * minmax(s')
    combined_risk_conflict  = 1 - combined_risk_isolation
    Returns a DataFrame and saves it as 'cow_combined_risk_scores_all_weeks.csv'.
    """
    def minmax(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return series * 0  # all zeros if no variation
        return (series - min_val) / (max_val - min_val)

    combined_risk_all = []
    for week_num in sorted(all_metrics_df['week'].unique()):
        week_df = all_metrics_df[all_metrics_df['week'] == week_num].copy()
        for pen_id, pen_df in week_df.groupby('pen_id'):
            pen_df = pen_df.copy()
            pen_df['s_prime'] = np.log1p(pen_df['strength'] / 3600)
            pen_df['degree_norm'] = minmax(pen_df['degree'])
            pen_df['s_prime_norm'] = minmax(pen_df['s_prime'])
            pen_df['conflict_risk'] = 0.4 * pen_df['degree_norm'] + 0.6 * pen_df['s_prime_norm']
            pen_df['isolation_risk'] = 1 - pen_df['conflict_risk']
            combined_risk_all.append(
                pen_df[['cow_id', 'week', 'pen_id', 'degree', 'strength', 's_prime', 'isolation_risk', 'conflict_risk']]
            )

    combined_risk_df = pd.concat(combined_risk_all, ignore_index=True)
    combined_risk_df.to_csv("cow_combined_risk_scores_all_weeks.csv", index=False)
    return combined_risk_df

# --------------------------
# Compute Temporal Deltas
# --------------------------
def compute_temporal_deltas(metrics_df, metric_cols=None):
    """
    Computes week-to-week deltas for each cow and pen for specified metric columns.
    Returns a DataFrame with columns: cow_id, pen_id, week, <metric>_delta
    """
    if metric_cols is None:
        metric_cols = ['degree', 'strength', 'betweenness', 'eigenvector']

    # Sort for correct diff calculation
    metrics_df = metrics_df.sort_values(['cow_id', 'pen_id', 'week'])

    # Compute deltas
    deltas = []
    for cow_id, cow_df in metrics_df.groupby('cow_id'):
        for pen_id, pen_df in cow_df.groupby('pen_id'):
            pen_df = pen_df.sort_values('week')
            for col in metric_cols:
                pen_df[f'{col}_delta'] = pen_df[col].diff().fillna(0)
            deltas.append(pen_df)

    return pd.concat(deltas, ignore_index=True)

# --------------------------
# Robust Z-Score
# --------------------------
def robust_zscore(series):
    median = series.median()
    iqr = series.quantile(0.75) - series.quantile(0.25)
    if iqr == 0:
        return (series - median)  # fallback
    return (series - median) / iqr

# --------------------------
# Collect Metrics Across Weeks
# --------------------------
all_metrics = []

for week_num in range(0, 4):  # weeks 0..3 in your data
    graphs = build_graphs(interactions, week_num)
    for pen_id, G in graphs.items():
        metrics = compute_metrics(G)
        metrics['pen_id'] = pen_id
        metrics['week'] = week_num
        all_metrics.append(metrics.reset_index().rename(columns={'index': 'cow_id'}))

all_metrics_df = pd.concat(all_metrics, ignore_index=True)

# Compute Temporal Deltas
temporal_deltas_df = compute_temporal_deltas(all_metrics_df)
temporal_deltas_df.to_csv("cow_temporal_deltas.csv", index=False)

# Compute and export combined risk scores for all weeks
combined_risk_df = compute_combined_risk(all_metrics_df)
combined_risk_df.to_csv("cow_combined_risk_scores.csv", index=False)

import seaborn as sns
import matplotlib.pyplot as plt

# Pivot the table: rows=cow_id, columns=week, values=isolation_risk
pivot = combined_risk_df.pivot_table(index='cow_id', columns='week', values='isolation_risk')

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, cmap='Reds', cbar_kws={'label': 'Isolation Risk'})
plt.title('Cow Isolation Risk Across Weeks')
plt.xlabel('Week')
plt.ylabel('Cow ID')
plt.tight_layout()
plt.show()