

# This file contain a Template class for mobility models.
# Mobility Models are used in order to create a DAG-instance defined in graph.py for
# further use in a decentralized CF algorithm (see ./src/algorithms/DecAlgoTemplate.py).

import math
import pandas as pd

class MobilityTemplate(object):
	""" Template Class that defines a framework to create DAGs accoring to some mobility model.
	Individual mobility models can be implemented by inheriting from this template class and
	substituting the <generate_graph> method.

	The mobility graph indicates when two nodes have a contact. The decentralized algorithms defined
	in DecAlgoTemplate will, for every time t and all nodes, collect the children. Then the parent node
	receives rating data from its children.

	Take for instance node A:1 in the below. A:1 only has C:0 as a child. Therefore, A will receive
	rating information from C which C had up until t=0. Vice versa, C:1 will receive rating information
	A had up until t=0.

	Example:
	############

	A:0		  B:0		C:0		^			|
	  \_________________/               |			|
	  /                 \               |			|
	A:1		  B:1		C:1		| flow direction	| time
	  \_______/                               |			|
	  /       \                               |			|
	A:2       B:2		C:2		|			V



	The basic variables comprise

	- graph			: <DAG>-instance (see ./src/mobility_models/graph.py)
	- N				: <int> fixed parameter to characterize the mobility model.
	- T				: <int> time horizon, the depth of the DAG. Either T or T_dict is None.
	- graph_seed	: <int> a random seed in case randomness is involved in the mobility model
	- T_dict		: <dic> a dictionary of {userId<str>:T_u<int>} pairs of userIds and time horizons T_u of the user.
	 				  if the T_dict is given, then the global time horizon T should be None. Individual time horizons T_u.

	The main function to call is the <generate_graph> function. When the function terminates, it has expanded the initial
	graph into an instance of the DAG class that holds the pairwise encounters between peers according to the mobility model.
	In cases in which the encounter is based on a random process. The graph_seed should be utilized in order to provide reproducibility
	and comparability.

	The <generate_graph> method in detail:

		I. Initialize all necessary information such as the future_snapshots_dict holding the current state of the snapshots (of all peers)
		II. Then iterate over all t = 1,..,T and all peers
			II.1 For_loop [can be parallelized over]
				II.1.1 Collect the payload [i.e. information to be shared by the children with the parent] over all children of a parent node
				II.1.2 Aggregate the information in the payload [if specified]
				II.1.3 Run a garbadge collection of the aggregated information in order to prevent for example duplicates
			II.2 Merge the aggregated information with the currently available information in the snapshot (future_snapshots_dict[peer])
			II.3 Every save_every_i periods, save all snapshots to a designated output_dir, where every future snapshot is saved to a distinct .csv file
				NOTE that the train_df is only needed at training time

		After the snapshots have been saved to disk, the functions in evaluation.py can be used in order to evaluate the goodness of the algorithm

	In order to use the template, the functions
		* II.1.1 collect_payload
		* II.1.2 aggregate_payload
		* II.1.3 collect_garbadge

	have to be implemented. The rest of the template should be used as is.


	Format:							  ALWAYS												ONLY if self.hide_p is not None
		self.__sim_dict:	keys	: [peer1+'-'+peer2]			 							OR	[peer1+'-'+peer2+':'+t]
							values	: similarity between initial_dfs of peer1 and peer2		similarity between peer1's initial_df and peer2's sampled profile
																							sampled profiles can be found in encounter_dict

		ONLY if self.hide_p is not None
		self.__encounter_dict:	keys	: [peer1+'-'+peer2+':'+t]
								values	: peer1's sampled profile for the edge (peer1:t, peer2:t-1) of the calculation graph (self.graph)



	"""
	# graph is a class instance of DAG
	def __init__(self, graph, N, T, graph_seed = None, T_dict = None):
		self.graph		= graph
		self.N			= N
		self.T			= T
		self.graph_seed	= graph_seed
		self.T_dict		= T_dict

	########################
	# 1. GETTER AND SETTER #
	########################
	def get_sim_dict(self):
		return self.sim_dict

	def set_sim_dict(self, new_sim_dict):
		self.sim_dict = new_sim_dict
		return





	#################################
	# 2. AUXILIARY FUNCTIONS        #
	#################################

	def sampleN(self, random_gen, N, peers, must_sample_list = None, drop_duplicates = False):
		""" Sample N distinct peers from a list of peers, unless the number of peers to sample from is smaller
			than N. In this case, return peers.
		Input:
			- random_gen		: instance of random.Random (for reproducible seeded random samples)
			- N					: <int> number of samples to draw from peers
			- peers				: <list> of elements to sample from (can be of any type)
			- must_sample_list	: <list> of elements that necessarily have to be contained in the output (sub)sample, else None
						  CAVEAT the elements in the must_sample_list will be included in the random_sample even if they are not included in the peers.
			- drop_duplicates	: <bool> if True, drop all duplicates in peers before sampling, else not.
		Output:
			- random_sample		: <list> of N elements of peers, if the number of elements in peers to sample from is at least as large as N
			- random_gen		: instance of random.Random

		"""
		# if drop_duplicates, remove all duplicates from the peers <list> before sampling
		if drop_duplicates:
			peers = pd.Series(peers).drop_duplicates().tolist()
		# trivial case of drawing N out of less than N items
		if len(peers)<N:
			return peers, random_gen
		# initialize list containing the random_sample
		if must_sample_list:
			random_sample = must_sample_list[:N]
			N -= len(must_sample_list)
		else:
			random_sample = []

		# for all (remaining) N
		for i in range(N):
			# sample one, and adjust the peers (list) to sample from
			sampled_peer, random_gen, peers = self.sample1(random_gen, peers)
			random_sample.append(sampled_peer)

		return random_sample, random_gen

	def sample1(self, random_gen, peers):
		""" Sample 1 distinct peer from a list of peers.
		Input:
			- random_gen		: instance of random.Random
			- peers				: <list> of elements to draw from
		Output:
			- sampled_peer		: The sampled single element of (original) peers
			- random_gen		:
			- remainder_peers	: <list> of the remainder of elements (original peers without sampled_peer)

		"""
		random_number = random_gen.random()
		# random_numer in [0.0, 1.0)
		num_peers = len(peers)
		random_number *= num_peers
		# random_number in [0.0, num_peers)
		random_index = math.floor(random_number)
		# select the corresponding peer
		sampled_peer = peers[random_index]
		# remove the sampled peer from the peers
		peers.pop(random_index)
		remainder_peers = peers
		return sampled_peer, random_gen, remainder_peers




	#####################################################
	# 3. GENERATE_GRAPH FUNCTIONS [MAIN FUNCTIONALITY]  #
	#####################################################
	def generate_graph(self):
		""" Build up the graph according some mobility model (assumptions).
		"""
