import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def plot_degree_distribution(G):
    print("\n--- Step 3: Degree Distribution Analysis ---")
    degrees = [G.degree(n) for n in G.nodes()]
    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), color='purple', alpha=0.7, edgecolor='black')
    plt.title("Node Degree Distribution")
    plt.xlabel("Degree (Number of Connections)")
    plt.ylabel("Count of Nodes")
    plt.savefig("degree_distribution.png")
    print("Degree distribution plot saved. Close window to continue...")
    plt.show()

def explain_model(model):
    print("\n--- Step 7: Explaining the Model")
    if hasattr(model, 'feature_importances_'):
        print("\n--- Step 11: Feature Importance (Explainable AI) ---")
        importances = model.feature_importances_
        feature_names = ['degree', 'pagerank', 'clustering']
        for name, imp in zip(feature_names, importances):
            print(f"{name.title()}: {imp:.4f}")

def visualize_network(G, df):
    print("\n--- Step 10: Visualizing the Network ---")
    plt.figure(figsize=(12, 8))
    
    pos = nx.spring_layout(G, seed=42)
    
    colors = ['skyblue' if G.nodes[n]['club'] == 'Mr. Hi' else 'salmon' for n in G.nodes()]
    
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=600, font_size=10, edge_color='gray', alpha=0.8)
    plt.title("Zachary's Karate Club: Node Affiliations (Blue: Mr. Hi, Red: Officer)")
    plt.savefig("network_visualization.png")
    print("Graph saved as 'network_visualization.png'")

    print("Opening visualization window...")
    plt.show()

def print_leaderboard(df):
    print("\n--- Step 9: Influence Leaderboard (Top 5) ---")
    leaderboard = df.sort_values(by='pagerank', ascending=False).head(5)
    print(leaderboard[['node', 'pagerank', 'actual_name']])

def compare_models(df):
    print("\n--- Step 6: Comparative Model Analysis ---")
    X = df[['degree', 'pagerank', 'clustering']]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)
    rf_score = rf.score(X_test, y_test)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_score = lr.score(X_test, y_test)

    print(f"Random Forest Accuracy: {rf_score:.2%}")
    print(f"Logistic Regression Accuracy: {lr_score:.2%}")
    
    return rf if rf_score >= lr_score else lr

def get_predictions(df, model):
    print("\n--- Step 8: Generating Predictions ---")
    X = df[['degree', 'pagerank', 'clustering']]
    df['predicted_club'] = model.predict(X)
    
    mapping = {1: 'Officer', 0: 'Mr. Hi'}
    df['actual_name'] = df['target'].map(mapping)
    df['predicted_name'] = df['predicted_club'].map(mapping)
    
    df['is_correct'] = df['actual_name'] == df['predicted_name']
    
    return df

def find_communities(G):
    print("\n--- Step 4: Community Detection ---")
    communities = nx.community.louvain_communities(G, seed=42)
    print(f"Detected {len(communities)} distinct communities within the club.")
    for i, comm in enumerate(communities):
        print(f"Community {i+1}: {list(comm)}")

def analyze_topology(G):
    print("\n--- Step 2: Global Topology Analysis ---")
    
    density = nx.density(G)
    avg_clustering = nx.average_clustering(G)
    
    random_G = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges())
    random_clustering = nx.average_clustering(random_G)
    
    print(f"Graph Density: {density:.4f}")
    print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
    print(f"Random Graph Clustering (Baseline): {random_clustering:.4f}")
    
    if avg_clustering > random_clustering:
        print("Insight: This network shows 'Small World' properties (High Clustering).")
    else:
        print("Insight: This network behaves like a random graph.")

def load_data():
    print("Step 1: Loading Internal Karate Club Data...")
    G = nx.karate_club_graph()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def extract_features(G):
    print("Step 5: Extracting structural features...")
    degree = nx.degree_centrality(G)      
    pagerank = nx.pagerank(G)            
    clustering = nx.clustering(G)        
    
    features = pd.DataFrame({
        'node': list(G.nodes()),
        'degree': [degree[n] for n in G.nodes()],
        'pagerank': [pagerank[n] for n in G.nodes()],
        'clustering': [clustering[n] for n in G.nodes()],
        'target': [1 if G.nodes[n]['club'] == 'Officer' else 0 for n in G.nodes()]
    })
    return features

def train_influence_model(df):
    print("Step 4: Training Model to Predict Club Affiliation...")
    
    X = df[['degree', 'pagerank', 'clustering']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred))
    return clf

if __name__ == "__main__":
    graph = load_data()
    analyze_topology(graph)
    plot_degree_distribution(graph)
    find_communities(graph)
    
    features_df = extract_features(graph)
    
    best_model = compare_models(features_df)
    
    # 4. Explanation
    explain_model(best_model)
    
    final_results = get_predictions(features_df, best_model)
    
    print("\n--- Final Prediction Table (All Nodes) ---")
    print(final_results[['node', 'actual_name', 'predicted_name', 'is_correct']].head(34))
    
    print_leaderboard(final_results)
    visualize_network(graph, final_results)
