#include "graph.hpp"

HGraph::HGraph() {
}


void HGraph::add_node(int n){
	for (unsigned int i=0; i<nodes.size(); i++ ) {
		if(nodes[i] == n){
			cout << "node " << n << " already exists." << endl;
			return;
		}
	}
	nodes.push_back(n);
	//	cout << "node at " << n << " created" << endl;
}

void HGraph::add_edge(edge e){
	for (unsigned int i=0; i<edges.size(); i++ ) {
		if(edges[i].from == e.from && edges[i].to == e.to){
			cout << "edge from " << e.from << " to" << e.to << " already exists." << endl;
			return;
		}
	}
	if(!node_exists(e.from))
		add_node(e.from);
	if(!node_exists(e.to))
		add_node(e.to);
	edges.push_back(e);
}

void HGraph::add_edge(int from, int to, double cost){
	edge e = {from, to, cost};
	for (unsigned int i=0; i<edges.size(); i++ ) {
		if(edges[i].from == e.from && edges[i].to == e.to){
			cout << "edge from " << e.from << " to" << e.to << " already exists." << endl;
			return;
		}
	}
	if(!node_exists(e.from))
		add_node(e.from);
	if(!node_exists(e.to))
		add_node(e.to);
	edges.push_back(e);
}

bool HGraph::node_exists(int n){
	for (unsigned int i=0; i<nodes.size(); i++ ) {
		if(nodes[i] == n){
			return true;
		}
	}
	return false;

}

bool HGraph::edge_exists(edge e){
	for (unsigned int i=0; i<edges.size(); i++ ) {
		if(edges[i].from == e.from && edges[i].to == e.to){
			return true;
		}
	}
	return false;
}

bool HGraph::edge_exists(int from, int to){
	edge e = {from, to, 0.0};
	for (unsigned int i=0; i<edges.size(); i++ ) {
		if(edges[i].from == e.from && edges[i].to == e.to){
			return true;
		}
	}
	return false;
}

void HGraph::print_nodes(){
	cout << "NODES:" << endl;
	for (unsigned int i=0; i<nodes.size(); i++ ) {
		cout << nodes[i] << ", ";
	}
	cout << endl;


}

void erase(vector<int>& vec, int ele){
	for (unsigned int i=0; i<vec.size(); i++ ) {
		if(vec[i] == ele){
			vec.erase(vec.begin() + i);
			return;
		}
	}
}

/*
 * This will give you the shortest path
 * from nodes[0] to nodes[last]
 * use h.path as path
 */

void HGraph::Dijkstra(vector<int>& path){
	cout << "Welcome to  Dijkstra!" << endl;

	// initialize some stuff
	double INF = 9999999.0;
	vector<dijk_pair> verts;
	for (unsigned int i=0; i<nodes.size(); i++ )
		verts.push_back({nodes[i], INF, false, -1});

	// get first node
	verts[0].dist = 0.0;
	int u=0;
	int v=0;
	bool run = true;

	cout << "Starting Dijkstra with " << verts.size() << " nodes." << endl;
	int count = 0;
	// Dijkstra Loop
	while (run){
//		cout << "Dijkstra " << count << " von " << verts.size() -1 << endl;
		// assign new node
		double d = INF+1;
		for (unsigned int i=0; i< verts.size(); i++){
			if(verts[i].visited == false && verts[i].dist < d){
				d = verts[i].dist;
				u = i;
			}
		}
		verts[u].visited = true;
		// check all edges and update dists
		for (unsigned int i=0; i< edges.size(); i++){
			if(edges[i].from == verts[u].node){
				for (unsigned int k=0; k< verts.size(); k++){
					if(edges[i].to == verts[k].node)
						v = k;
				}
				double alt = verts[u].dist + edges[i].cost;
				//				cout << "Edge " << verts[u].node << " " << verts[v].node << " " <<  edges[i].cost << " alt: " << alt << endl;
				if (alt < verts[v].dist){
					verts[v].dist = alt;
					verts[v].previous = verts[u].node;
				}
			}
		}
		// return if there are no more unvisited nodes
		run = false;
		for (unsigned int i=0; i< verts.size(); i++){
			if (verts[i].visited == false){
				run = true;
			}
		}
		count++;
	}

//	cout << "Nun nur noch backwards path!" << endl;


	// print whole matrix
	for (unsigned int i=0; i<verts.size(); i++ ) {
		//		cout << nodes[i] << "; " << verts[i].dist << "; " << verts[i].previous << "; " << verts[i].visited <<  endl;
	}



	// Pick shortest path from matrix
	int curr = verts.size() - 1;
	path.push_back(verts[curr].node);
	while(true){
		if(curr == nodes[0] ||  verts[curr].previous == -1)
			break;
//		cout << "Node is: " << verts[curr].node << " previous is: " << verts[curr].previous << endl;
		path.push_back(verts[curr].previous);
		for (unsigned int i=0; i<verts.size(); i++ ){
			if (verts[curr].previous == verts[i].node)
				curr = i;

		}
	}
	path.push_back(verts[0].node);

	// reverse path
	reverse(path.begin(), path.end());

	cout << "Es wurden " << path.size() << " Frames selektiert." << endl;
		for (unsigned int i=0; i<path.size(); i++ ) {
			cout <<  path[i] << "," << endl;
		}
}




