import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

# ---------- Utility Functions ----------

def sigmoid(x):
    """Convert z-score to 0-1 probability scale"""
    return 1 / (1 + math.exp(-np.clip(x, -10, 10)))  # Clip to prevent overflow

def robust_z_score(values, per_pen=True, pen_ids=None):
    """
    Calculate robust z-scores using median and IQR instead of mean and std
    Formula: z_robust = (x - median) / (IQR / 1.35)
    """
    if per_pen and pen_ids:
        pen_groups = {}
        for cow_id, score in values.items():
            pen = pen_ids.get(cow_id)
            if pen not in pen_groups:
                pen_groups[pen] = {}
            pen_groups[pen][cow_id] = score
        normalized_scores = {}
        for pen, pen_values in pen_groups.items():
            scores_array = np.array(list(pen_values.values()))
            median = np.median(scores_array)
            q75 = np.percentile(scores_array, 75)
            q25 = np.percentile(scores_array, 25)
            iqr = q75 - q25
            if iqr == 0:
                iqr = np.std(scores_array) * 1.35
                if iqr == 0:
                    iqr = 1.0
            for cow_id, score in pen_values.items():
                z_robust = (score - median) / (iqr / 1.35)
                normalized_scores[cow_id] = z_robust
    else:
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

def build_weekly_graph(interactions_df, week, pen_id, graph_cache):
    """Build or retrieve NetworkX graph for specific week and pen"""
    cache_key = (week, pen_id)
    if cache_key in graph_cache:
        return graph_cache[cache_key]
    week_data = interactions_df[
        (interactions_df['week'] == week) &
        (interactions_df['pen_id'] == pen_id)
    ]
    G = nx.Graph()
    for _, row in week_data.iterrows():
        cow_i, cow_j = row['cow_i'], row['cow_j']
        duration = (row['end_ts'] - row['start_ts'])
        if G.has_edge(cow_i, cow_j):
            G[cow_i][cow_j]['weight'] += duration
            G[cow_i][cow_j]['count'] += 1
        else:
            G.add_edge(cow_i, cow_j, weight=duration, count=1)
    graph_cache[cache_key] = G
    return G

# ---------- Component Calculations (Optimized with Caching) ----------

def calc_all_centralities(G, centrality_cache, cache_key):
    """Compute and cache all centralities for a graph"""
    if cache_key in centrality_cache:
        return centrality_cache[cache_key]
    if len(G.nodes()) == 0:
        centrality_cache[cache_key] = {
            'betweenness': {},
            'degree': {},
            'closeness': {}
        }
        return centrality_cache[cache_key]
    centrality_cache[cache_key] = {
        'betweenness': nx.betweenness_centrality(G, normalized=True),
        'degree': nx.degree_centrality(G),
        'closeness': nx.closeness_centrality(G)
    }
    return centrality_cache[cache_key]

def calc_betweenness_centrality(centralities, cow_id):
    return centralities['betweenness'].get(cow_id, 0)

def calc_degree_centrality(centralities, cow_id):
    return centralities['degree'].get(cow_id, 0)

def calc_closeness_centrality(centralities, cow_id):
    return centralities['closeness'].get(cow_id, 0)

def calc_degree_change_score(degree_centrality_cache, cow_id, current_week, previous_week, pen_id):
    if previous_week is None:
        return 0.0
    curr = degree_centrality_cache.get((current_week, pen_id), {}).get(cow_id, 0)
    prev = degree_centrality_cache.get((previous_week, pen_id), {}).get(cow_id, 0)
    if prev == 0:
        return 0.0
    change = (curr - prev) / prev
    return sigmoid(change)

def calc_community_disruption(G, cow_id, community_cache, cache_key):
    """Calculate how much a cow bridges between communities"""
    if cow_id not in G.nodes() or len(G.nodes()) < 3:
        return 0.0
    if cache_key in community_cache:
        communities = community_cache[cache_key]
    else:
        try:
            communities = list(nx.community.greedy_modularity_communities(G))
            community_cache[cache_key] = communities
        except:
            community_cache[cache_key] = []
            return 0.0
    cow_community = None
    for i, community in enumerate(communities):
        if cow_id in community:
            cow_community = i
            break
    if cow_community is None:
        return 0.0
    neighbors = list(G.neighbors(cow_id))
    if not neighbors:
        return 0.0
    cross_community_connections = 0
    for neighbor in neighbors:
        neighbor_community = None
        for i, community in enumerate(communities):
            if neighbor in community:
                neighbor_community = i
                break
        if neighbor_community is not None and neighbor_community != cow_community:
            cross_community_connections += 1
    return cross_community_connections / len(neighbors)

def calc_interaction_frequency_deviation(interactions_df, cow_id, week, pen_id, freq_cache):
    cache_key = (week, pen_id)
    if cache_key in freq_cache:
        cow_interaction_counts, pen_avg_interactions = freq_cache[cache_key]
    else:
        pen_data = interactions_df[
            (interactions_df['week'] == week) & 
            (interactions_df['pen_id'] == pen_id)
        ]
        cow_interaction_counts = {}
        for _, row in pen_data.iterrows():
            cow_i, cow_j = row['cow_i'], row['cow_j']
            cow_interaction_counts[cow_i] = cow_interaction_counts.get(cow_i, 0) + 1
            cow_interaction_counts[cow_j] = cow_interaction_counts.get(cow_j, 0) + 1
        pen_avg_interactions = np.mean(list(cow_interaction_counts.values())) if cow_interaction_counts else 0
        freq_cache[cache_key] = (cow_interaction_counts, pen_avg_interactions)
    cow_interactions = cow_interaction_counts.get(cow_id, 0)
    if pen_avg_interactions > 0:
        frequency_deviation = abs(cow_interactions - pen_avg_interactions) / pen_avg_interactions
        return min(1.0, frequency_deviation)
    return 0.0

def calc_centrality_decline_score(degree_centrality_cache, cow_id, weeks, pen_id):
    if len(weeks) < 2:
        return 0.0
    centrality_scores = []
    sorted_weeks = sorted(weeks)
    for week in sorted_weeks:
        centrality_scores.append(degree_centrality_cache.get((week, pen_id), {}).get(cow_id, 0))
    if len(centrality_scores) < 2:
        return 0.0
    x = np.arange(len(centrality_scores))
    try:
        slope, _, _, _, _ = stats.linregress(x, centrality_scores)
        centrality_decline_score = max(0, -slope)
        return min(1.0, centrality_decline_score * 10)
    except:
        return 0.0

# ---------- Main Risk Calculation with Robust Z-Scores (Optimized) ----------

def calculate_risk_scores_with_robust_normalization(interactions_df, pen_assignments_df):
    """
    Calculate conflict and isolation risks with robust z-score normalization
    Returns: pd.DataFrame with columns: week, pen_id, cow_id, conflict_risk, isolation_risk
    """
    risk_records = []
    weeks = sorted(pen_assignments_df['week'].unique())
    # Caches for optimization
    graph_cache = {}
    centrality_cache = {}
    community_cache = {}
    freq_cache = {}
    degree_centrality_cache = {}

    # Precompute all centralities and degree_centrality for all (week, pen)
    for week in weeks:
        for pen_id in pen_assignments_df[pen_assignments_df['week'] == week]['pen_id'].unique():
            G = build_weekly_graph(interactions_df, week, pen_id, graph_cache)
            cache_key = (week, pen_id)
            centralities = calc_all_centralities(G, centrality_cache, cache_key)
            degree_centrality_cache[cache_key] = centralities['degree']

    # Store raw component scores for robust normalization
    all_components = {
        'betweenness': {},
        'degree_change': {},
        'community_disruption': {},
        'frequency_deviation': {},
        'degree_deficit': {},
        'centrality_decline': {},
        'closeness_deficit': {}
    }
    cow_pen_mapping = {}

    # First pass: Calculate raw component scores
    for week in weeks:
        previous_week = week - 1 if week - 1 in weeks else None
        week_assignments = pen_assignments_df[pen_assignments_df['week'] == week]
        for _, row in week_assignments.iterrows():
            cow_id = row['cow_id']
            pen_id = row['pen_id']
            cow_pen_mapping[f"{cow_id}_{week}"] = pen_id
            cache_key = (week, pen_id)
            centralities = centrality_cache[cache_key]
            # Conflict risk components
            betweenness = calc_betweenness_centrality(centralities, cow_id)
            degree_change = calc_degree_change_score(degree_centrality_cache, cow_id, week, previous_week, pen_id)
            G = graph_cache[cache_key]
            community_disruption = calc_community_disruption(G, cow_id, community_cache, cache_key)
            frequency_deviation = calc_interaction_frequency_deviation(
                interactions_df, cow_id, week, pen_id, freq_cache
            )
            # Isolation risk components
            degree = calc_degree_centrality(centralities, cow_id)
            degree_deficit = 1 - degree
            weeks_for_decline = [w for w in weeks if w <= week]
            centrality_decline = calc_centrality_decline_score(
                degree_centrality_cache, cow_id, weeks_for_decline, pen_id
            )
            closeness = calc_closeness_centrality(centralities, cow_id)
            closeness_deficit = 1 - closeness
            # Store raw scores
            cow_week_key = f"{cow_id}_{week}"
            all_components['betweenness'][cow_week_key] = betweenness
            all_components['degree_change'][cow_week_key] = degree_change
            all_components['community_disruption'][cow_week_key] = community_disruption
            all_components['frequency_deviation'][cow_week_key] = frequency_deviation
            all_components['degree_deficit'][cow_week_key] = degree_deficit
            all_components['centrality_decline'][cow_week_key] = centrality_decline
            all_components['closeness_deficit'][cow_week_key] = closeness_deficit

    # Second pass: Apply robust z-score normalization per pen
    normalized_components = {}
    for component_name, raw_scores in all_components.items():
        normalized_components[component_name] = robust_z_score(
            raw_scores, 
            per_pen=True, 
            pen_ids=cow_pen_mapping
        )

    # Third pass: Calculate final risk scores using normalized components
    for week in weeks:
        week_assignments = pen_assignments_df[pen_assignments_df['week'] == week]
        for _, row in week_assignments.iterrows():
            cow_id = row['cow_id']
            pen_id = row['pen_id']
            cow_week_key = f"{cow_id}_{week}"
            # Get normalized components (convert z-scores to 0-1 scale)
            btw_norm = sigmoid(normalized_components['betweenness'].get(cow_week_key, 0))
            deg_ch_norm = sigmoid(normalized_components['degree_change'].get(cow_week_key, 0))
            comm_norm = sigmoid(normalized_components['community_disruption'].get(cow_week_key, 0))
            freq_norm = sigmoid(normalized_components['frequency_deviation'].get(cow_week_key, 0))
            deg_def_norm = sigmoid(normalized_components['degree_deficit'].get(cow_week_key, 0))
            cen_decl_norm = sigmoid(normalized_components['centrality_decline'].get(cow_week_key, 0))
            clo_def_norm = sigmoid(normalized_components['closeness_deficit'].get(cow_week_key, 0))
            # Calculate weighted risk scores
            conflict_risk = (
                0.50 * btw_norm +
                0.25 * deg_ch_norm +
                0.15 * comm_norm +
                0.10 * freq_norm
            )
            isolation_risk = (
                0.35 * deg_def_norm +
                0.30 * cen_decl_norm +
                0.20 * clo_def_norm +
                0.15 * freq_norm
            )
            risk_records.append({
                'week': week,
                'pen_id': pen_id,
                'cow_id': cow_id,
                'conflict_risk': min(1.0, conflict_risk),
                'isolation_risk': min(1.0, isolation_risk),
                # Store normalized components for analysis
                'betweenness_norm': btw_norm,
                'degree_change_norm': deg_ch_norm,
                'community_disruption_norm': comm_norm,
                'frequency_deviation_norm': freq_norm,
                'degree_deficit_norm': deg_def_norm,
                'centrality_decline_norm': cen_decl_norm,
                'closeness_deficit_norm': clo_def_norm
            })
    return pd.DataFrame(risk_records)

# ---------- Visualization Functions ----------

def plot_risk_heatmap(risk_df):
    """Create heatmap visualization of risk scores"""
    conflict_pivot = risk_df.pivot(index='cow_id', columns='week', values='conflict_risk')
    isolation_pivot = risk_df.pivot(index='cow_id', columns='week', values='isolation_risk')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    sns.heatmap(conflict_pivot, annot=True, fmt='.2f', cmap='Reds', 
                ax=ax1, cbar_kws={'label': 'Conflict Risk'})
    ax1.set_title('Conflict Risk Scores by Cow and Week')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Cow ID')
    sns.heatmap(isolation_pivot, annot=True, fmt='.2f', cmap='Blues', 
                ax=ax2, cbar_kws={'label': 'Isolation Risk'})
    ax2.set_title('Isolation Risk Scores by Cow and Week')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Cow ID')
    plt.tight_layout()
    plt.show()
    return fig

def plot_risk_trends(risk_df):
    """Plot risk score trends over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    for cow_id in risk_df['cow_id'].unique():
        cow_data = risk_df[risk_df['cow_id'] == cow_id]
        ax1.plot(cow_data['week'], cow_data['conflict_risk'], 
                marker='o', label=f'Cow {cow_id}', alpha=0.7)
    ax1.set_title('Conflict Risk Trends Over Time')
    ax1.set_xlabel('Week')
    ax1.set_ylabel('Conflict Risk Score')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, 
                label='High Risk Threshold (0.7)')
    for cow_id in risk_df['cow_id'].unique():
        cow_data = risk_df[risk_df['cow_id'] == cow_id]
        ax2.plot(cow_data['week'], cow_data['isolation_risk'], 
                marker='s', label=f'Cow {cow_id}', alpha=0.7)
    ax2.set_title('Isolation Risk Trends Over Time')
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Isolation Risk Score')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.7, color='blue', linestyle='--', alpha=0.5,
                label='High Risk Threshold (0.7)')
    plt.tight_layout()
    plt.show()
    return fig

def generate_risk_summary_table(risk_df):
    """Generate summary statistics table"""
    summary_stats = risk_df.groupby('cow_id').agg({
        'conflict_risk': ['mean', 'std', 'max'],
        'isolation_risk': ['mean', 'std', 'max']
    }).round(3)
    summary_stats.columns = [f"{col[1]}_{col[0]}" for col in summary_stats.columns]
    summary_stats['conflict_high_risk'] = summary_stats['max_conflict_risk'] > 0.7
    summary_stats['isolation_high_risk'] = summary_stats['max_isolation_risk'] > 0.7
    return summary_stats

# ---------- Main Execution Function ----------

def main():
    print("Loading data...")
    interactions_df = pd.read_csv('interactions.csv')
    if 'week' not in interactions_df.columns:
        SECONDS_PER_WEEK = 604800
        interactions_df['week'] = (interactions_df['start_ts'].astype(int) // SECONDS_PER_WEEK).astype(int)
    pen_assignments_df = pd.read_csv('pen_assignments.csv')
    cow_registry_df = pd.read_csv('cow_registry.csv')
    pen_map = pen_assignments_df[['week', 'cow_id', 'pen_id']]
    interactions_df = interactions_df.merge(
        pen_map.rename(columns={'cow_id': 'cow_i', 'pen_id': 'pen_id'}),
        on=['week', 'cow_i'], how='left'
    )
    interactions_df = interactions_df.merge(
        pen_map.rename(columns={'cow_id': 'cow_j', 'pen_id': 'pen_j'}),
        on=['week', 'cow_j'], how='left'
    )
    interactions_df = interactions_df[interactions_df['pen_id'] == interactions_df['pen_j']]
    print("Calculating risk scores with robust z-score normalization...")
    risk_scores_df = calculate_risk_scores_with_robust_normalization(
        interactions_df, pen_assignments_df
    )
    print("\nRisk Scores Summary:")
    print("=" * 50)
    summary_table = generate_risk_summary_table(risk_scores_df)
    print(summary_table)
    risk_scores_df.to_csv('risk_scores_robust_normalized.csv', index=False)
    summary_table.to_csv('risk_summary_statistics.csv', index=True)
    print(f"\nResults saved to:")
    print("- risk_scores_robust_normalized.csv")
    print("- risk_summary_statistics.csv")
    print("\nGenerating visualizations...")
    heatmap_fig = plot_risk_heatmap(risk_scores_df)
    heatmap_fig.savefig('risk_heatmap.png', dpi=300, bbox_inches='tight')
    trends_fig = plot_risk_trends(risk_scores_df)
    trends_fig.savefig('risk_trends.png', dpi=300, bbox_inches='tight')
    print("Visualizations saved:")
    print("- risk_heatmap.png")
    print("- risk_trends.png")
    high_risk_cows = risk_scores_df[
        (risk_scores_df['conflict_risk'] > 0.7) | 
        (risk_scores_df['isolation_risk'] > 0.7)
    ]
    if not high_risk_cows.empty:
        print(f"\nHigh-Risk Cows Identified ({len(high_risk_cows)} instances):")
        print("=" * 60)
        for _, cow in high_risk_cows.iterrows():
            risk_type = []
            if cow['conflict_risk'] > 0.7:
                risk_type.append(f"Conflict: {cow['conflict_risk']:.3f}")
            if cow['isolation_risk'] > 0.7:
                risk_type.append(f"Isolation: {cow['isolation_risk']:.3f}")
            print(f"Week {cow['week']}, Pen {cow['pen_id']}, "
                  f"Cow {cow['cow_id']}: {', '.join(risk_type)}")
    return risk_scores_df, summary_table

if __name__ == '__main__':
    risk_scores, summary = main()