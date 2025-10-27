import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

# ---------- Utility Functions ----------

def sigmoid(x):
    return 1 / (1 + math.exp(-np.clip(x, -10, 10)))

def robust_z_score(values):
    """Robust z-score normalization (whole herd)."""
    if not values:
        return {}
    scores_array = np.array(list(values.values()))
    median = np.median(scores_array)
    q75 = np.percentile(scores_array, 75)
    q25 = np.percentile(scores_array, 25)
    iqr = q75 - q25
    if iqr == 0:
        iqr = np.std(scores_array) * 1.35
        if iqr == 0:
            iqr = 1.0
    normalized_scores = {}
    for cow_id, score in values.items():
        z_robust = (score - median) / (iqr / 1.35)
        normalized_scores[cow_id] = z_robust
    return normalized_scores

def build_weekly_graph(interactions_df, week, graph_cache):
    """Build NetworkX graph for a given week (all cows same pen)."""
    if week in graph_cache:
        return graph_cache[week]
    week_data = interactions_df[interactions_df['week'] == week]
    G = nx.Graph()
    for _, row in week_data.iterrows():
        cow_i, cow_j = row['cow_i'], row['cow_j']
        duration = (row['end_ts'] - row['start_ts'])
        if G.has_edge(cow_i, cow_j):
            G[cow_i][cow_j]['weight'] += duration
            G[cow_i][cow_j]['count'] += 1
        else:
            G.add_edge(cow_i, cow_j, weight=duration, count=1)
    graph_cache[week] = G
    return G

def calc_all_centralities(G, centrality_cache, week):
    if week in centrality_cache:
        return centrality_cache[week]
    if len(G.nodes()) == 0:
        centrality_cache[week] = {'betweenness': {}, 'degree': {}, 'closeness': {}}
        return centrality_cache[week]
    centrality_cache[week] = {
        'betweenness': nx.betweenness_centrality(G, normalized=True),
        'degree': nx.degree_centrality(G),
        'closeness': nx.closeness_centrality(G)
    }
    return centrality_cache[week]

def calc_degree_change_score(degree_centrality_cache, cow_id, current_week, previous_week):
    if previous_week is None:
        return 0.0
    curr = degree_centrality_cache.get(current_week, {}).get(cow_id, 0)
    prev = degree_centrality_cache.get(previous_week, {}).get(cow_id, 0)
    if prev == 0:
        return 0.0
    return sigmoid((curr - prev) / prev)

def calc_community_disruption(G, cow_id, community_cache, week):
    if cow_id not in G.nodes() or len(G.nodes()) < 3:
        return 0.0
    if week in community_cache:
        communities = community_cache[week]
    else:
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            community_cache[week] = communities
        except:
            return 0.0
    cow_community = None
    for i, comm in enumerate(communities):
        if cow_id in comm:
            cow_community = i
            break
    if cow_community is None:
        return 0.0
    neighbors = list(G.neighbors(cow_id))
    if not neighbors:
        return 0.0
    cross_connections = sum(
        1 for n in neighbors 
        if any(n in comm for i, comm in enumerate(communities) if i != cow_community)
    )
    return cross_connections / len(neighbors)

def calc_interaction_frequency_deviation(interactions_df, cow_id, week, freq_cache):
    if week in freq_cache:
        cow_counts, avg = freq_cache[week]
    else:
        week_data = interactions_df[interactions_df['week'] == week]
        cow_counts = {}
        for _, row in week_data.iterrows():
            cow_counts[row['cow_i']] = cow_counts.get(row['cow_i'], 0) + 1
            cow_counts[row['cow_j']] = cow_counts.get(row['cow_j'], 0) + 1
        avg = np.mean(list(cow_counts.values())) if cow_counts else 0
        freq_cache[week] = (cow_counts, avg)
    cow_interactions = cow_counts.get(cow_id, 0)
    if avg > 0:
        return min(1.0, abs(cow_interactions - avg) / avg)
    return 0.0

def calc_centrality_decline_score(degree_centrality_cache, cow_id, weeks):
    if len(weeks) < 2:
        return 0.0
    scores = [degree_centrality_cache.get(w, {}).get(cow_id, 0) for w in weeks]
    x = np.arange(len(scores))
    try:
        slope, _, _, _, _ = stats.linregress(x, scores)
        return min(1.0, max(0, -slope) * 10)
    except:
        return 0.0

# ---------- Risk Score Calculation ----------

def calculate_risk_scores(interactions_df):
    risk_records = []
    weeks = sorted(interactions_df['week'].unique())
    graph_cache, centrality_cache, community_cache, freq_cache, degree_cache = {}, {}, {}, {}, {}

    # Precompute centralities
    for w in weeks:
        G = build_weekly_graph(interactions_df, w, graph_cache)
        centralities = calc_all_centralities(G, centrality_cache, w)
        degree_cache[w] = centralities['degree']

    # Raw component storage
    all_components = {k: {} for k in [
        'betweenness','degree_change','community_disruption',
        'frequency_deviation','degree_deficit','centrality_decline','closeness_deficit'
    ]}

    # First pass
    for w in weeks:
        prev_w = w - 1 if (w - 1) in weeks else None
        G = graph_cache[w]
        centralities = centrality_cache[w]
        for cow_id in list(G.nodes()):
            betweenness = centralities['betweenness'].get(cow_id, 0)
            degree_change = calc_degree_change_score(degree_cache, cow_id, w, prev_w)
            comm_disrupt = calc_community_disruption(G, cow_id, community_cache, w)
            freq_dev = calc_interaction_frequency_deviation(interactions_df, cow_id, w, freq_cache)
            degree = centralities['degree'].get(cow_id, 0)
            degree_deficit = 1 - degree
            decline = calc_centrality_decline_score(degree_cache, cow_id, [x for x in weeks if x <= w])
            closeness = centralities['closeness'].get(cow_id, 0)
            closeness_deficit = 1 - closeness
            key = f"{cow_id}_{w}"
            all_components['betweenness'][key] = betweenness
            all_components['degree_change'][key] = degree_change
            all_components['community_disruption'][key] = comm_disrupt
            all_components['frequency_deviation'][key] = freq_dev
            all_components['degree_deficit'][key] = degree_deficit
            all_components['centrality_decline'][key] = decline
            all_components['closeness_deficit'][key] = closeness_deficit

    # Normalize
    norm_components = {k: robust_z_score(v) for k,v in all_components.items()}

    # Final risk
    for w in weeks:
        G = graph_cache[w]
        for cow_id in list(G.nodes()):
            key = f"{cow_id}_{w}"
            btw = sigmoid(norm_components['betweenness'].get(key, 0))
            deg_ch = sigmoid(norm_components['degree_change'].get(key, 0))
            comm = sigmoid(norm_components['community_disruption'].get(key, 0))
            freq = sigmoid(norm_components['frequency_deviation'].get(key, 0))
            deg_def = sigmoid(norm_components['degree_deficit'].get(key, 0))
            decl = sigmoid(norm_components['centrality_decline'].get(key, 0))
            clo_def = sigmoid(norm_components['closeness_deficit'].get(key, 0))
            conflict = 0.50*btw + 0.25*deg_ch + 0.15*comm + 0.10*freq
            isolation = 0.35*deg_def + 0.30*decl + 0.20*clo_def + 0.15*freq
            risk_records.append({
                'week': w,
                'cow_id': cow_id,
                'conflict_risk': min(1.0, conflict),
                'isolation_risk': min(1.0, isolation)
            })
    return pd.DataFrame(risk_records)

# ---------- Visualization ----------

def plot_risk_heatmaps(risk_df):
    for risk_type in ['conflict_risk', 'isolation_risk']:
        pivot = risk_df.pivot(index='cow_id', columns='week', values=risk_type)
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={'label': risk_type})
        plt.title(f"{risk_type.capitalize()} Heatmap (by Week)")
        plt.ylabel("Cow ID")
        plt.xlabel("Week")
        plt.tight_layout()
        plt.savefig(f"{risk_type}_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

def plot_risk_trends(risk_df):
    for risk_type in ['conflict_risk', 'isolation_risk']:
        plt.figure(figsize=(10, 6))
        for cow_id in risk_df['cow_id'].unique():
            cow_data = risk_df[risk_df['cow_id'] == cow_id]
            plt.plot(cow_data['week'], cow_data[risk_type], marker='o', label=f"Cow {cow_id}")
        plt.title(f"{risk_type.capitalize()} Trends Over Time")
        plt.xlabel("Week")
        plt.ylabel("Risk Score")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{risk_type}_trend.png", dpi=300, bbox_inches='tight')
        plt.close()

# ---------- Main ----------

def main():
    interactions_df = pd.read_csv("interactions.csv")
    if 'week' not in interactions_df.columns:
        SECONDS_PER_WEEK = 604800
        interactions_df['week'] = (interactions_df['start_ts'].astype(int) // SECONDS_PER_WEEK).astype(int)

    risk_df = calculate_risk_scores(interactions_df)
    print(risk_df.head())
    risk_df.to_csv("risk_scores_single_pen.csv", index=False)

    # Visualization
    plot_risk_heatmaps(risk_df)
    plot_risk_trends(risk_df)

if __name__ == "__main__":
    main()