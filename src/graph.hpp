#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include<stdio.h>
#include<iostream>
#include <vector>
#include <algorithm>
using namespace std;

class HGraph {


public:
	struct edge{
		int from;
		int to;
		double cost;
	};

	struct dijk_pair{
		int node;
		double dist;
		bool visited;
		int previous;
	};
	HGraph();
	vector<int> nodes;
	vector<edge> edges;


	void add_node(int n);
	void add_edge(edge e);
	void add_edge(int from, int to, double cost);

	bool node_exists(int n);
	bool edge_exists(edge e);
	bool edge_exists(int from, int to);
	void print_nodes();

	void Dijkstra(vector<int>& path);




};

#endif /* GRAPH_HPP_ */
