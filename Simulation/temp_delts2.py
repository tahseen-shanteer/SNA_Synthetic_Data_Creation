import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import math

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_preprocess_data():
    """Load and preprocess the data"""
    # Load your data files
    interactions = pd.read_csv('interactions.csv')
    pen_assignments = pd.read_csv('pen_assignments.csv')
    
    # Convert seconds to weeks (604800 seconds/week)
    interactions['week'] = (interactions['start_ts'] / 604800).astype(int)
    
    return interactions, pen_assignments

def calculate_weekly_centrality(interactions, pen_assignments):
    """Calculate centrality metrics for each pen per week"""
    all_centrality_data = []
    
    # Get unique weeks and pens
    unique_weeks = sorted(interactions['week'].unique())
    unique_pens = pen_assignments['pen_id'].unique()
    
    for week_num, pen_id in product(unique_weeks, unique_pens):
        try:
            # Get cows in pen for that week
            pen_cows = pen_assignments[
                (pen_assignments['week'] == week_num) & 
                (pen_assignments['pen_id'] == pen_id)
            ]['cow_id'].unique()
            
            if len(pen_cows) == 0:
                continue
                
            # Get interactions for those cows in that week
            weekly_interactions = interactions[
                (interactions['week'] == week_num) &
                (interactions['cow_i'].isin(pen_cows)) &
                (interactions['cow_j'].isin(pen_cows))
            ]
            
            if len(weekly_interactions) == 0:
                # Create empty centrality entries for weeks with no interactions
                for cow_id in pen_cows:
                    all_centrality_data.append({
                        'week': week_num,
                        'pen_id': pen_id,
                        'cow_id': cow_id,
                        'degree': 0,
                        'betweenness': 0,
                        'eigenvector': 0,
                        'interaction_count': 0
                    })
                continue
            
            # Create graph
            G = nx.Graph()
            for _, row in weekly_interactions.iterrows():
                duration = row['end_ts'] - row['start_ts']
                if G.has_edge(row['cow_i'], row['cow_j']):
                    G[row['cow_i']][row['cow_j']]['weight'] += duration
                else:
                    G.add_edge(row['cow_i'], row['cow_j'], weight=duration)
            
            # Add isolated nodes (cows with no interactions)
            for cow_id in pen_cows:
                if cow_id not in G.nodes():
                    G.add_node(cow_id)
            
            # Calculate centrality metrics with error handling
            centrality_data = {}
            
            # Degree centrality
            centrality_data['degree'] = nx.degree_centrality(G)
            
            # Betweenness centrality (handle disconnected graphs)
            try:
                centrality_data['betweenness'] = nx.betweenness_centrality(G, weight='weight', normalized=True)
            except:
                centrality_data['betweenness'] = {node: 0 for node in G.nodes()}
            
            # Eigenvector centrality (handle convergence issues)
            try:
                centrality_data['eigenvector'] = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1e-6)
            except:
                try:
                    centrality_data['eigenvector'] = nx.eigenvector_centrality(G, max_iter=2000, tol=1e-4)
                except:
                    centrality_data['eigenvector'] = {node: 0 for node in G.nodes()}
            
            # Store results
            for cow_id in pen_cows:
                all_centrality_data.append({
                    'week': week_num,
                    'pen_id': pen_id,
                    'cow_id': cow_id,
                    'degree': centrality_data['degree'].get(cow_id, 0),
                    'betweenness': centrality_data['betweenness'].get(cow_id, 0),
                    'eigenvector': centrality_data['eigenvector'].get(cow_id, 0),
                    'interaction_count': len([edge for edge in G.edges(cow_id)])
                })
                
        except Exception as e:
            print(f"Error processing week {week_num}, pen {pen_id}: {e}")
            continue
    
    return pd.DataFrame(all_centrality_data)

def compute_weekly_deltas(centrality_df):
    """Compute weekly deltas for centrality metrics"""
    delta_data = []
    
    # Group by pen and cow
    grouped = centrality_df.groupby(['pen_id', 'cow_id'])
    
    for (pen_id, cow_id), group in grouped:
        group = group.sort_values('week')
        
        # Calculate deltas
        for metric in ['degree', 'betweenness', 'eigenvector']:
            group[f'{metric}_delta'] = group[metric].diff()
        
        delta_data.append(group)
    
    return pd.concat(delta_data, ignore_index=True)

def plot_all_cows_centrality(centrality_df, delta_df):
    """Create individual plots for ALL cows, organized by pen"""
    
    unique_pens = centrality_df['pen_id'].unique()
    
    for pen_id in unique_pens:
        print(f"Creating plots for Pen {pen_id}...")
        
        # Get cows in this pen
        pen_cows = centrality_df[centrality_df['pen_id'] == pen_id]['cow_id'].unique()
        
        # Calculate grid size for subplots
        n_cows = len(pen_cows)
        if n_cows == 0:
            continue
            
        n_cols = min(4, n_cows)  # Maximum 4 columns
        n_rows = math.ceil(n_cows / n_cols)
        
        # Create figure for centrality values
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig1.suptitle(f'Centrality Metrics - Pen {pen_id}', fontsize=16, y=1.02)
        
        # Create figure for deltas
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig2.suptitle(f'Weekly Centrality Deltas - Pen {pen_id}', fontsize=16, y=1.02)
        
        # Flatten axes for easy indexing
        if n_rows == 1 and n_cols == 1:
            axes1 = [axes1]
            axes2 = [axes2]
        else:
            axes1 = axes1.flatten() if axes1.ndim > 1 else axes1
            axes2 = axes2.flatten() if axes2.ndim > 1 else axes2
        
        metrics = ['degree', 'betweenness', 'eigenvector']
        colors = ['blue', 'green', 'red']
        
        for idx, cow_id in enumerate(pen_cows):
            if idx >= len(axes1):
                break
                
            # Get data for this cow
            cow_centrality = centrality_df[
                (centrality_df['cow_id'] == cow_id) & 
                (centrality_df['pen_id'] == pen_id)
            ].sort_values('week')
            
            cow_delta = delta_df[
                (delta_df['cow_id'] == cow_id) & 
                (delta_df['pen_id'] == pen_id)
            ].sort_values('week')
            
            # Plot centrality values
            for i, metric in enumerate(metrics):
                axes1[idx].plot(cow_centrality['week'], cow_centrality[metric], 
                               color=colors[i], marker='o', linewidth=2, label=metric)
            
            axes1[idx].set_title(f'Cow {cow_id}')
            axes1[idx].set_xlabel('Week')
            axes1[idx].set_ylabel('Centrality Value')
            axes1[idx].legend()
            axes1[idx].grid(True, alpha=0.3)
            
            # Plot deltas
            for i, metric in enumerate(metrics):
                axes2[idx].plot(cow_delta['week'], cow_delta[f'{metric}_delta'], 
                               color=colors[i], marker='s', linewidth=2, label=f'Δ {metric}')
            
            axes2[idx].set_title(f'Cow {cow_id} - Deltas')
            axes2[idx].set_xlabel('Week')
            axes2[idx].set_ylabel('Δ Centrality Value')
            axes2[idx].legend()
            axes2[idx].grid(True, alpha=0.3)
            axes2[idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Hide empty subplots
        for idx in range(len(pen_cows), len(axes1)):
            axes1[idx].set_visible(False)
            axes2[idx].set_visible(False)
        
        plt.tight_layout()
        fig1.savefig(f'pen_{pen_id}_centrality.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'pen_{pen_id}_deltas.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)

def plot_summary_statistics(centrality_df, delta_df):
    """Plot summary statistics across all pens"""
    
    # 1. Average centrality trends by pen
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    metrics = ['degree', 'betweenness', 'eigenvector']
    
    for i, metric in enumerate(metrics):
        # Average centrality values
        avg_centrality = centrality_df.groupby(['week', 'pen_id'])[metric].mean().reset_index()
        
        for pen_id in avg_centrality['pen_id'].unique():
            pen_data = avg_centrality[avg_centrality['pen_id'] == pen_id]
            axes[i, 0].plot(pen_data['week'], pen_data[metric], 
                           marker='o', label=f'Pen {pen_id}', linewidth=2)
        
        axes[i, 0].set_title(f'Average {metric.title()} Centrality by Pen')
        axes[i, 0].set_ylabel(f'{metric.title()} Centrality')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Average deltas
        avg_delta = delta_df.groupby(['week', 'pen_id'])[f'{metric}_delta'].mean().reset_index()
        
        for pen_id in avg_delta['pen_id'].unique():
            pen_data = avg_delta[avg_delta['pen_id'] == pen_id]
            axes[i, 1].plot(pen_data['week'], pen_data[f'{metric}_delta'], 
                           marker='s', label=f'Pen {pen_id}', linewidth=2)
        
        axes[i, 1].set_title(f'Average Weekly Δ {metric.title()} Centrality')
        axes[i, 1].set_ylabel(f'Δ {metric.title()} Centrality')
        axes[i, 1].set_xlabel('Week')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('summary_centrality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Heatmap of degree centrality changes
    pivot_data = delta_df.pivot_table(index='week', columns='pen_id', 
                                     values='degree_delta', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Average Degree Centrality Delta'})
    plt.title('Weekly Changes in Degree Centrality by Pen')
    plt.tight_layout()
    plt.savefig('centrality_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run the entire analysis"""
    print("Loading data...")
    interactions, pen_assignments = load_and_preprocess_data()
    
    print("Calculating centrality metrics...")
    centrality_df = calculate_weekly_centrality(interactions, pen_assignments)
    
    print("Computing weekly deltas...")
    delta_df = compute_weekly_deltas(centrality_df)
    
    print("Creating individual cow plots...")
    plot_all_cows_centrality(centrality_df, delta_df)
    
    print("Creating summary plots...")
    plot_summary_statistics(centrality_df, delta_df)
    
    # Save results to CSV
    centrality_df.to_csv('weekly_centrality_metrics.csv', index=False)
    delta_df.to_csv('weekly_centrality_deltas.csv', index=False)
    
    print("Analysis complete! Files saved:")
    print("- weekly_centrality_metrics.csv")
    print("- weekly_centrality_deltas.csv")
    print("- pen_XX_centrality.png (for each pen)")
    print("- pen_XX_deltas.png (for each pen)")
    print("- summary_centrality_analysis.png")
    print("- centrality_heatmap.png")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Time period: {centrality_df['week'].min()} to {centrality_df['week'].max()} weeks")
    print(f"Number of pens: {centrality_df['pen_id'].nunique()}")
    print(f"Number of cows: {centrality_df['cow_id'].nunique()}")
    print(f"Total observations: {len(centrality_df)}")

# Run the program
if __name__ == "__main__":
    main()