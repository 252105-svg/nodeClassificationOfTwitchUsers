# Social Network Analysis: Predictive Modeling in Zachary's Karate Club

## Project Overview
This project implements a multi-stage Social Network Analysis (SNA) and Machine Learning pipeline. The goal is to predict the eventual affiliation of club members during a famous institutional split using only structural graph metrics.

## Dataset
We utilize **Zachary’s Karate Club**, a foundational sociological dataset. It represents the social ties between 34 members of a karate club that split into two factions: one led by the instructor (**Mr. Hi**) and the other by the administrator (**The Officer**).

## The 13-Step Analytical Pipeline
The project is developed iteratively through Git, following these logical phases:

1.  **Data Ingestion**: Loading the graph topology.
2.  **Global Topology**: Analyzing density and "Small World" properties.
3.  **Statistical Analysis**: Visualizing the Power-Law degree distribution.
4.  **Community Detection**: Identifying natural tribes using the Louvain algorithm.
5.  **Feature Engineering**: Calculating Degree, PageRank, and Clustering coefficients.
6.  **Model Comparison**: Benchmarking Random Forest vs. Logistic Regression.
7.  **Explainable AI (XAI)**: Extracting feature importance to understand "Social Power."
8.  **Predictive Modeling**: Generating faction affiliation for every node.
9.  **Influence Ranking**: Identifying the most prestigious members (PageRank).
10. **Network Visualization**: Force-directed graph plots colored by affiliation.
11. **Bridge Detection**: Finding "Gatekeeper" nodes via Betweenness Centrality.
12. **Correlation Analysis**: Studying the redundancy of social metrics.
13. **Robustness Testing**: Simulating a targeted attack on the network hubs.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Execute the pipeline: `python main.py`

## Key Findings
- The network exhibits high clustering, confirming it is not a random graph.
- Structural features (Degree and PageRank) provide nearly 100% accuracy in predicting social behavior.
- Targeted removal of the primary hub (Node 0) significantly alters the network's connectivity.
