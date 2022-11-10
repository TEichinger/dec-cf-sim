from collections import deque

from collections.abc import Iterable # for python >= 3.6

import pandas as pd

from copy import deepcopy

class DAG(object):
	"""
	Description:
	############
	Class (DAG = directed acyclic graph) that represents the encounters of single individuals in a P2P network.
	The graph represents a DAG where the keys also contain temporal information. E.g node A at time t = 0 is denoted
	"A:0". The graph itself is a (big) dictionary that provides fast (O(1)) lookups whenever the keys are known.
	The search of subgraphs is linear in the sum of degrees along the path down-stream (direction of the edges is down).

	Example:
	########
	A DAG represented by a dictionary, where keys represent nodes of the DAG and values directed edges.
	The flow direction is 'upward': This factilitates search for sub-DAGs.

	This example dictionary...

	{ "A:0":             deque([])
	  "B:0":             deque([])
	  "C:0":             deque([])
	  "A:1": deque(["A:0", "C:0"])
	  "B:1":        deque(["B:0"])
	  "C:1": deque(["A:0", "C:0"])
	}

	... represents the following DAG ...

	B:0    A:0   C:0     ^
	|       | \_/ |      |
	|       | / \ |      | flow direction
	B:1    A:1   C:1     |

	"""

	def __init__(self, graph_dict=None, peers=[], T = None, T_dict = None):
		"""
        	Initialize a graph object. When no graph dictionary is provided, initialize an empty graph.
			- self.__graph_dict		: <dict> with the format as in the Example
			- self.__peers			: <list> a list of users independent of time, e.g. ["A", "B", "C"] in the Example
			- self.__T			: <int> time horizon, T = 1 in the Example. This is equal to the depth of the DAG.
		"""
		if graph_dict is None:
			graph_dict = {}
		self.__graph_dict = graph_dict

		self.__peers	  = peers
		self.__T          = T

		self.mobility_model = None

	#######################################################
	# 0. Return functions                                 #
	#######################################################

	def graph_dict(self):
		""" Return the graph_dict specified in the graph."""
		return self.__graph_dict
	def peers(self):
		""" Return the peers specified in the graph. """
		return self.__peers

	def T(self):
		""" Return T. """
		return self.__T

	def T_of_peer(self, peer):
		""" Return the time horizon of a peer, that is the height of the subgraph
		that only contains the nodes corresponding to a peer.

		Example:
		########

			B:0    A:0   C:0     ^
			          \_/        |
			         / \        | flow direction
			B:1    A:1   C:1     |
			   \_/  			 |
			   / \       		 |
			B:2    A:2   C:2	 |

			T_A = 2
			T_B = 2
			T_C = 1

			For A and B, the for loop (X) does never enter the if (XX), which means that the maximum number of epochs <peer_nodes> is the peer's time horizon <T_of_peer>.
			For C though, the if (XX) is entered for peer_node == C:2 as C:2 does not have any children. The time horizon <T_of_peer> is thus 1.
		"""
		# select the nodes in the graph that correspond to <peer> (<peer> is left to the colon)
		# For example peer == "1"
		# Then ["1:3", "3:1"] -> ["1:3"]
		peer_nodes = [peer_node for peer_node in self.__graph_dict.keys() if (peer == peer_node.split(":")[0]) & (int(peer_node.split(":")[1]) > 0)]

		epochs_of_peer_nodes_with_children = [int(peer_node.split(':')[1]) for peer_node in peer_nodes if len(self.children(peer_node))>0]
		T_of_peer = max(epochs_of_peer_nodes_with_children) if epochs_of_peer_nodes_with_children != [] else 0

		return T_of_peer


	def vertices(self):
		""" Return the vertices of the graph. Max. 100"""
		vertices = list(self.__graph_dict.keys())
		return vertices

	def edges(self):
		""" Return the edges of the graph as a list of duples. """
		return self.__collect_edges()

	def __collect_edges(self):
		""" Collect the edges given in the self.graph_dict. This method is only invoked by self.edges."""
		edges = []
		for parent in self.__graph_dict.keys():
			for child in self.__graph_dict[parent]:
				if (parent, child) not in edges:
					edges.append((parent, child))
		return edges

	def children(self, vertex):
		""" Return the children of a vertex. """
		return self.__graph_dict[vertex]

	#######################################################
	# 1. Update Functions                                 #
	#######################################################
	def update_T(self):
		""" Update the time horizon T.
		This is useful if vertices and edges have been added and the depth of the DAG is not equal to self.__T anymore.
		"""
		vertices = self.vertices()
		t_s = [int(el.split(":")[1]) for el in vertices if ":" in el]
		self.__T = max(t_s)

	def initialize_peers(self, t = 0):
		""" Initialize a set of vertices of all <self.__peers> at time <t>=t.
		Input:
			- t:	<int>

		"""
		for peer in self.__peers:
			vertex_key = peer + ":" + str(t)
			self.add_vertex(vertex_key)

	def use_mobility_model(self, mobility_model):
		self.mobility_model = mobility_model

	#######################################################
	# 2. Graph Building Functions                         #
	#######################################################

	def add_vertex(self, vertex):
		""" If the vertex 'vertex' is not in self.__graph_dict, a key 'vertex' with empty value is added.
        Otherwise nothing has to be done
        requires:
            - vertex (<str>): key under which the added (empty) vertex should be searchable
		"""
		if vertex not in self.__graph_dict:
			self.__graph_dict[vertex] = deque()

	def add_edge(self, source, target):
		""" An ordered duple (<list> or <tuple> type) of vertex keys (source,target). """
		if source in self.__graph_dict.keys():
			self.__graph_dict[source].append(target)
		else:
			self.__graph_dict[source] = deque([target])

		if target not in self.__graph_dict.keys():
			self.add_vertex(target)


	def collect_history(self, vertex, timestamp_delta = None):
		""" For a vertex (e.g. some peer at time t --> "peer:t"), collect all values for the keys
		"peer:t", "peer:t-1", "peer:t-2", .., "peer:0", if timestamp_delta = None. The values represent the nodes' children.
		Else collect all values for the keys
		"peer:t", "peer:t-1", .., "peer:t-d+1"

		NOTE that an empty list is reported, when the vertex does not have a history.

		collect_history.Example
		#######################
		This example dictionary...

		{ "A:0":             deque([])
		  "B:0":             deque([])
		  "A:1": deque(["A:0", "B:0"])
		  "B:1": deque(["A:0", "B:0"])
		}

		... represents the following DAG ...

		A:0   B:0     ^
		 | \_/ |      |
		 | / \ |      | flow direction
		A:1   B:1     |

		Then self.collect_history("A:1") yields ['A:0', 'B:0']
		and  self.collect_history("A:0") yields []

		Input:
			- vertex	 : <str>
			- timestamp_delta: <int> number of epochs back in time to consider fetching the history from (default: None --> all is fetched)
		Output:
			- result	 : <list> of vertex string-identifiers
		"""
		result = []
		peer =     vertex.split(":")[0]
		t    = int(vertex.split(":")[1])

		if timestamp_delta is None:
			start_timestamp = 0
		else:
			start_timestamp = max(0,t-timestamp_delta+1)
		# for i = t, t-1, .., 1, 0
		for i in reversed(range(start_timestamp,t+1,1)):
			result += self.__graph_dict[peer + ":" + str(i)]
		return result

	#######################################################
	# 3. Graph Altering Functions                         #
	#######################################################

	def prune_by_t(self, peers, peer_Ts, new_peer_Ts):#peer, min_t, max_t, rewire = False):
		""" Prune edges to nodes of <peer> between min_t <= t <= max_t.

			* mobility_model : instance of the MobilityTemplate abstract base class; e.g. an instance of the <AssignNMobility> class
								If specified, the <edges_to_peer_to_remove> are rewired according to the <mobility_model.generate_graph_at_t> method.
								More specifically with respect the below example. We see that pruning causes the edge ('A:2', 'C:1') to be removed.
								Although peer 'A' might be interested in establishing a contact to another peer, the current execution graph does not allow that.
								As 'B' is also interested in sharin ratings in epoch 2 (see that there is an edge ('B:2', 'B:1')), rewiring should
								add two edges ('B:2', 'A:1') and ('A:2', 'B:1') in order to compensate for the removal of edge ('A:2', 'C:1').



			NOTE that no vertices are removed!

		 		Example:
		 		########

		 			B:0    A:0    C:0     ^
		 			 | \_/  |      |      |
		 			 | / \  |      |      | flow direction
		 			B:1    A:1    C:1     |
		 			 |      |  \_/ |	  |
		 			 |      |  / \ |      |
		 			B:2    A:2    C:2	  |

					##### prune_by_T('C', 1, 2) #####
					yields

		 			B:0    A:0    C:0     ^
		 			 | \_/  |      |      |
		 			 | / \  |      |      | flow direction
		 			B:1    A:1    C:1     |
		 			 |      |        	  |
		 			 |      |             |
		 			B:2    A:2    C:2	  |


		"""
		# CANCEL CRITERION
		#########################################################################
		# if peer_Ts and new_peer_Ts are identical, nothing is to be done!
		if peer_Ts == new_peer_Ts:
			return
		#
		#########################################################################
		edges = self.edges() # example: [('A:1', 'A:0'), ('A:1', 'C:0'), ('B:1', 'B:0'), ('C:1', 'A:0'), ('C:1', 'C:0'), ('C:2', 'A:1'), ('C:2', 'C:1'), ('A:2'), ('C:1')]

		edges_from_peer_to_remove = []
		edges_to_peer_to_remove   = []
		# I. Remove all edges from and to a <peer> for t > <new_peer_T> [<peer> will not enter into contact past <new_peer_T>]
		#################################################################
		# for all peers
		for peer, peer_T, new_peer_T in zip(peers, peer_Ts, new_peer_Ts):
			# if the new parameter value differs from the current parameter value
			if peer_T > new_peer_T:
				# select edges to remove -> <edges_to_remove>
				# the left if statement handles parents (edge[0]) and the right statement handles children (edge[1])
				# for the example the upper  statement removes the edges ('C:2', 'A:1'), ('C:2', 'C:1')
				#                 the bottom statement removes the edge ('A:2'), ('C:1')
				edges_from_peer_to_remove += [edge for edge in edges if ((edge[0].split(":")[0] == peer) & (int(edge[0].split(":")[1]) >  new_peer_T ))]
				edges_to_peer_to_remove   += [edge for edge in edges if ((edge[1].split(":")[0] == peer) & (int(edge[1].split(":")[1]) >= new_peer_T ))]

		# put together all edges to remove
		edges_to_remove = edges_from_peer_to_remove + edges_to_peer_to_remove

		# and remove those edges from the graph
		for edge_to_remove in edges_to_remove:
			source, target = edge_to_remove
			self.remove_edge(source,target)

		# II. Recreate edges that were removed in (I.) from users who lost a connection (= <edges_to_peer_to_remove>)
		###################################################################
		# rewire the <edges_to_peer_to_remove> if a mobility_model is specified
		if (self.mobility_model is not None) & (self.mobility_model.mobility_string == "AssignN"):
			# define a pandas dataframe with peer indices and a single column named "counter".
			# The counter column defines the maximum number of edges to obtain per peer.
			# Example: In the following example we have three peers 'A', 'B', and 'C' who should establish a connection to at most 2 other users.
			#		 	counter
			#		A	2
			#		B	2
			#		C	2

			sampled_edges = []
			# sort edges in <edges_to_peer_to_remove> by epoch and select epochs
			sorted_epochs = sorted(list(set([int(edge[0].split(":")[1]) for edge in edges_to_peer_to_remove])))

			# for every epoch <t> that requires recreation of lost edeges
			# EXAMPLE: min(new_peer_Ts) = 3 and max(peer_Ts) = 5, then epochs <sorted_epochs> are strictly larger than min(new_peer_Ts)
			for t in sorted_epochs:
				# II.1 Create a peer_counter_df that holds the number of edges to recreate per user
				#####################################################################################
				# initialize a peer_counter_df with zeros
				peer_counter_df = pd.DataFrame([0 for _ in peers], index = peers, columns = ["counter"])
				# for every peer
				for peer in peers:
					new_peer_T = new_peer_Ts[self.__peers.tolist().index(peer)]
					# pass if epoch <t> > <new_peer_T> [<peer> does not seek connections]
					if t > new_peer_T:
						pass
					else:
						# lookup the number of edges at epoch t (that remain after removing edges in (I.))
						num_edges_of_peer_at_t = len(self.children(peer+":"+str(t)))
						# add the difference between the global number of connections that a peer seeks in an epoch <self.mobility_model.N> and
						# the current number of edges <num_edges_of_peer_at_t>
						peer_counter_df.loc[peer,"counter"] += self.mobility_model.N - num_edges_of_peer_at_t

				# sample edges (and vertices) to add to the graph
				sampled_edges_t, _ = self.mobility_model.sample_edges_and_vertices_at_t(peer_counter_df, t, self.mobility_model.random_gen)
				sampled_edges += sampled_edges_t

			# add sampled edges
			for edge in sampled_edges:
				if edge not in self.edges():
					self.add_edge(edge[0], edge[1])

		return

	def augment_by_t(self, peers, peer_Ts, new_peer_Ts):
		""" Add edges to nodes of <peer> if ther current time horizon T is smaller than the newly calculated time horizon.
			For instance <DistributedGradientTracking> may yield an increase in a peer's time horizon. In this case, peers need to add edges to
			the already existing execution graph.

		* mobility_model : instance of the MobilityTemplate abstract base class; e.g. an instance of the <AssignNMobility> class
							If specified, the <edges_to_peer_to_remove> are rewired according to the <mobility_model.generate_graph_at_t> method.
							More specifically with respect the below example. We see that pruning causes the edge ('A:2', 'C:1') to be removed.
							Although peer 'A' might be interested in establishing a contact to another peer, the current execution graph does not allow that.
							As 'B' is also interested in sharin ratings in epoch 2 (see that there is an edge ('B:2', 'B:1')), rewiring should
							add two edges ('B:2', 'A:1') and ('A:2', 'B:1') in order to compensate for the removal of edge ('A:2', 'C:1').



		NOTE that no vertices are removed!

			Example:
			########

				B:0    A:0    C:0     ^
				 | \_/  |      |      |
				 | / \  |      |      | flow direction
				B:1    A:1    C:1     |
				 |      |  \_/ |	  |
				 |      |  / \ |      |
				B:2    A:2    C:2	  |

				##### prune_by_T('C', 1, 2) #####
				yields

				B:0    A:0    C:0     ^
				 | \_/  |      |      |
				 | / \  |      |      | flow direction
				B:1    A:1    C:1     |
				 |      |        	  |
				 |      |             |
				B:2    A:2    C:2	  |


		"""



	def remove_edge(self, source, target):
		""" Remove the edge between the source node <source> and the target node <target>, provided it exists. """
		if source in self.__graph_dict.keys():
			if target in self.__graph_dict[source]:
				self.__graph_dict[source].remove(target)
		return



def main():
	graph_dict = 	{ "A:0":             deque([]),
			  "B:0":             deque([]),
			  "C:0":             deque([]),
			  "A:1": deque(["A:0", "C:0"]),
			  "B:1":        deque(["B:0"]),
			  "C:1": deque(["A:0", "C:0"]),
			  "C:2": deque(["A:1", "C:1"]),
			  "A:2": deque(["C:1"])
			}

	#graph_dict2 = { "A:0":             deque([])}


	dag = DAG(graph_dict, peers = ["A", "B", "C"])
	#dag2 = DAG(graph_dict2)
	print(dag.edges())
	dag.prune_by_t("C", 1, 2)#remove_edge("C:1", "C:0")
	print(dag.edges())

	#dag2.add_vertex("A:1")
	#dag2.add_edge("A:1", "A:0")

	#print(dag2.edges())



	"""
	dag.set_snapshot("A:0", "test")

	vertices = dag.vertices()
	edges = dag.edges()
	snapshot0 = dag.snapshot("A:0")
	snapshot1 = dag.snapshot("A:1")
	print("Vertices: {}".format(vertices))
	print("Edges: {}".format(edges))

	print("Snapshot at A:0 : {}".format(snapshot0))
	print("Snapshot at A:1 : {}".format(snapshot1))

	dag.initialize_peers()
	print(dag.vertices())
	"""
if __name__ == "__main__":
	main()
