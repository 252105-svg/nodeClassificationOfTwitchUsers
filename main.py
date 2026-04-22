from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_influence_model(df):
    print("\n--- Step 3: Machine Learning ---")
    
    threshold = df['degree'].quantile(0.8)
    df['target'] = (df['degree'] > threshold).astype(int)
    
    X = df[['pagerank', 'clustering']]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return clf

if __name__ == "__main__":
    graph = load_data()
    features_df = extract_features(graph)
    model = train_influence_model(features_df) 
