#include <iostream>
#include <fstream>
#include <vector>
#include <queue>
#include <string>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <map>

using namespace std;

struct Graph {
    int n;
    vector<vector<pair<int, double>>> adj;
};

map<int, int> id2node, node2id;

// Updated loadGraph: assumes each line in the file contains an edge "u v p"
Graph loadGraph(const string &filename) {
    ifstream fin(filename);
    if (!fin) {
        cerr << "Error opening graph file: " << filename << endl;
        exit(1);
    }
    int u, v, u_id, v_id;
    double p;
    int maxNode = 0;
    vector<tuple<int, int, double>> edges;
    
    // Read each edge and determine the maximum node id.
    while (fin >> u >> v >> p) {
        if (node2id.find(u) == node2id.end()){
            node2id[u] = maxNode;
            id2node[maxNode] = u;
            maxNode++;
        }
        u_id = node2id[u];
        if (node2id.find(v) == node2id.end()){
            node2id[v] = maxNode;
            id2node[maxNode] = v;
            maxNode++;
        }
        v_id = node2id[v];
        edges.push_back(make_tuple(u_id, v_id, p));
    }
    fin.close();

    Graph g;
    g.n = maxNode + 1;
    g.adj.resize(g.n);
    
    // Populate the adjacency list.
    for (auto &edge : edges) {
        tie(u_id, v_id, p) = edge;
        g.adj[u_id].push_back({v_id, p});
    }
    
    return g;
}

// Simulate infection spread starting from 'start' over random_instances simulations.
// This version reuses a preallocated "visited" vector and a queue buffer to avoid repeated allocations.
double simulateSpread(int start, const Graph &g, const vector<bool>& removed, 
                      vector<double>& probCounts, int random_instances, 
                      mt19937 &rng) {
    vector<int> infectionCount(g.n, 0);
    double totalSpread = 0.0;
    uniform_real_distribution<double> dist(0.0, 1.0);

    // Preallocate a visited buffer once.
    vector<bool> visited(g.n, false);
    // Preallocate a queue buffer.
    queue<int> q;
    
    for (int instance = 0; instance < random_instances; instance++) {
        if (removed[start])
            break;
        
        // Reset the visited vector.
        fill(visited.begin(), visited.end(), false);
        // Clear the queue by reinitializing it.
        q = queue<int>();
        
        visited[start] = true;
        q.push(start);
        while (!q.empty()) {
            int u_id = q.front();
            q.pop();
            for (auto &edge : g.adj[u_id]) {
                int v_id = edge.first;
                double p = edge.second;
                if (removed[v_id] || visited[v_id])
                    continue;
                double r = dist(rng);
                if (r <= p) {
                    visited[v_id] = true;
                    q.push(v_id);
                }
            }
        }
        int spread = 0;
        for (int i = 0; i < g.n; i++) {
            if (visited[i]) {
                infectionCount[i]++;
                spread++;
            }
        }
        totalSpread += spread;
    }
    probCounts.resize(g.n, 0.0);
    for (int i = 0; i < g.n; i++) {
        probCounts[i] = static_cast<double>(infectionCount[i]) / random_instances;
    }
    return totalSpread / random_instances;
}

// Iterates over all candidate nodes (those not removed) and returns the node with maximum expected spread.
// The corresponding infection probability vector is returned in bestProbabilities.
int chooseBestNode(const Graph &g, const vector<bool>& removed, int random_instances, 
                   vector<double>& bestProbabilities, mt19937 &rng, double &bestSpread) {
    double tempSpread = -1.0;
    int bestNode = -1;
    vector<double> tempProbabilities;
    for (int node = 0; node < g.n; node++) {
        if (removed[node])
            continue;
        double spread = simulateSpread(node, g, removed, tempProbabilities, random_instances, rng);
        if (spread > tempSpread) {
            tempSpread = spread;
            bestNode = node;
            bestProbabilities = tempProbabilities;
        }
    }
    bestSpread = tempSpread;
    return bestNode;
}

// Marks nodes as removed if their infection probability is greater than the threshold.
// The bestNode (seed) is also removed.
void removeInfectedNodes(const vector<double>& infectionProbabilities, double threshold, 
                         vector<bool>& removed, int bestNode) {
    for (size_t i = 0; i < infectionProbabilities.size(); i++) {
        if (infectionProbabilities[i] > threshold || (int)i == bestNode) {
            removed[i] = true;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <graph_file> <output_file> <k> <random_instances>" << endl;
        return 1;
    }
    
    string graphFile = argv[1];
    string outputFile = argv[2];
    int k = stoi(argv[3]);
    // int random_instances = stoi(argv[4]);
    int random_instances = 100;
    
    Graph g = loadGraph(graphFile);
    vector<bool> removed(g.n, false); // Initially, no nodes are removed.
    
    // Open output file for incremental writing.
    ofstream fout(outputFile);
    if (!fout) {
        cerr << "Error opening output file: " << outputFile << endl;
        return 1;
    }
    
    // Hyperparameter: obtained after hyperparameter tuning on the two datasets
    double infectivityThreshold = 0.06;
    
    random_device rd;
    mt19937 rng(rd());
    
    // Greedily choose k seed nodes.
    for (int i = 0; i < k; i++) {
        vector<double> bestProbabilities;
        double bestSpread;
        int bestNodeId = chooseBestNode(g, removed, random_instances, bestProbabilities, rng, bestSpread);
        if (bestNodeId == -1) {
            cerr << "No candidate found in iteration " << i << ". Ending selection." << endl;
            break;
        }
        int bestNode = id2node[bestNodeId];
        // Immediately write the selected node to the output file.
        fout << bestNode << "\n";
        fout.flush();  // flush to ensure the node is written immediately.
        
        cerr << "Iteration " << i << ": Selected seed node " << bestNode 
             << " with expected spread " << bestSpread << endl;
        
        // Remove nodes highly likely to be infected by this seed.
        removeInfectedNodes(bestProbabilities, infectivityThreshold, removed, bestNode);
        
        // Debug: count remaining unremoved candidate nodes.
        int remaining = 0;
        for (bool flag : removed) {
            if (!flag)
                remaining++;
        }
        cerr << "Remaining candidate nodes after removal: " << remaining << endl;
    }
    
    fout.close();
    return 0;
}
