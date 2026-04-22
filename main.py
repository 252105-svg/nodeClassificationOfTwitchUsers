import networkx as nx
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data():
    print("Step 1: Loading Internal Karate Club Data...")
    G = nx.karate_club_graph()
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

def extract_features(G):
    print("Step 2: Extracting structural features...")
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
    print("Step 3: Training Model to Predict Club Affiliation...")
    
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
    features_df = extract_features(graph)
    model = train_influence_model(features_df)
