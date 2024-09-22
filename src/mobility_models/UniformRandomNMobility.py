
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)

# for debugging
from utilities.util import load_ratings_df, make_train_test_split, build_execution_graph

# for production
import pandas as pd
import random
import time
from itertools import product
import multiprocessing
from multiprocessing import Pool, Manager, Process

from mobility_models.graph import DAG
from mobility_models.MobilityTemplate import MobilityTemplate
from utilities.util import UpperTriangleGenerator, UpperTriangleGeneratorX, UpperTriangleGeneratorY, RandomBinaryIterator,  split_every_n


class UniformRandomNMobility(MobilityTemplate):
	""" Class that defines a mobility model that assigns (N<=num peers) distinct other peers in the network to every peer.
	In contrast to AssignNMobility, the links between peers are not bi-direction, which means that a peer "A" might be
	linked to "B", yet not the other way round.

	NOTE:	Future nodes of one single peer are not connected with each other.
			So for instance, no edges as : "peer:3" --> "peer:2"; "peer:2" --> "peer:1" ...

	NOTE:	Although the principle of this mobility model is similar to that of Erdös-Rényi random graphs, it is different in that every
			peer necessarily has N links to other peers.

	Example:
	########

	N 		= 1
	T		= 1
	graph_seed	= 1234
	peers		= ['A', 'B', 'C']
	graph		= DAG(peers=peers)
	mobility_model = UniformRandomNMobility(graph, N, T, graph_seed = graph_seed)
	# generate the execution graph
	execution_graph = build_execution_graph(peers, mobility_model)

	Then the execution graph is represented by the following directed edges

	A:1 --> C:0
	B:1 --> A:0
	C:1 --> A:0
	A:2 --> C:1
	B:2 --> C:1
	C:2 --> B:1


	"""
	#################################
	# 1. INITIALIZATION FUNCTIONS   #
	#################################

	def __init__(self, graph, N, T, graph_seed = None):
		# call the template's initilization
		super().__init__(graph = graph, N = N, T = T, graph_seed = graph_seed)
		# mobility string for future reference
		self.mobility_string	= "UniformRandomN"


	##########################################
	# 3. GENERATE GRAPH [MAIN FUNCTIONALITY] #
	##########################################

	def generate_graph(self):
		""" Build up the graph such that every peer is assigned N other peers in the network. In comparison to UniformRandomMobility,
		the generation of this mobility graph scales linearly, since not all pairwise encounters have to be rolled (uniformly randomly).
		In order to assign random peers, we use the <> method from random. Links are NOT bi-directional.

		"""
		# set peer_counter_dict that for every peer, counts the number of times he/she has already been drawn randomly
		peers = self.graph.peers()

		# initialize a list of counters/dataframe of counters
		#    counter
		# 0     0
		# 1     0
		# ..    0
		peer_counter_df = pd.DataFrame([0 for _ in peers], index = peers, columns = ["counter"])

		# initialize peers at time t=0
		self.graph.initialize_peers()

		# initialize a random series of integers (generator object) that creates a unique series of integers
		graph_seed = self.graph_seed
		random_gen  = random.Random(graph_seed)


		# for all time periods
		for i in range(self.T):
			# for every peer1 in peers [peer1 is linked with peers, differently put, the peers in peer2_list are peer1's view in the P2P network
			for peer1 in peers:
				# sample N peers from the peers
				sampling_list = [el for el in peers if el != peer1]
				peer2_list, random_gen = self.sampleN(random_gen, self.N , sampling_list)
				# (*) link peer1 with all peer2's
				for peer2 in peer2_list:
					# link peer1 time i+1 with peer2 time id [target sends data to source]
					source_key = peer1+":"+str(i+1)
					target_key = peer2+":"+str(i)
					self.graph.add_edge(source_key, target_key)




if __name__ == "__main__":
	N 		= 1               # size of the neighborhoods
	T		= 2		  # number of epochs
	graph_seed	= 1234		  # for graph generation

	peers = ['A', 'B', 'C']
	graph			= DAG(peers=peers)
	# initialize the mobility model
	mobility_model = UniformRandomNMobility(graph, N, T, graph_seed = graph_seed)
	# generate the execution graph
	execution_graph = build_execution_graph(peers, mobility_model)

	print(execution_graph.edges())
	print(execution_graph.vertices())
