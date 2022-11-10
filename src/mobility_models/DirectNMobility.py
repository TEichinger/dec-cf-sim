
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)
# for debugging
from utilities.util import load_ratings_df, make_train_test_split, build_execution_graph

import random

# for production
from mobility_models.MobilityTemplate import MobilityTemplate
from mobility_models.graph import DAG
from utilities.util import UpperTriangleGenerator
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class DirectNMobility(MobilityTemplate):
	""" Class that defines a mobility model that assigns peers to N other similar (!) peers per epoch on average.
		Two users are considered similar if their cosine similarity is larger than the similarity threshold <self.min_sim>.

		NOTE We assume that similarity is symmetric. That is sim(a,b) = sim(b,a) for some peers a and b.


	Input:
		- train_df		: <pandas.DataFrame> a training dataframe to extract pairwise similarities from
						  train_df should have coordinate-format, that is include columns ["userId", "itemId", "rating"]
						  similarities will be calculated on the basis of user-profiles (that is ratings of a user)
		- N				: <int> number of contacts per user and epoch. Note that not necessarily every user is paired with N other users,
						  since some users only have few similar other users.
		- sim_string	: <str> string identifier that is associated with a specific similarity measure; e.g. "cosine" for
						  cosine similarity. See (X)
		- min_sim		: <float> similarity threshold only above which to consider peers as neighbors; set -1.0, if you
						  want to consider all peers. NOTE that this might cause results to be un-reproducible, if for instance
						  multiple candidate neighbors have the same similarity (e.g. 0.0)

	Example:
	########

	The train_df given in the test-file (./data/test) will produce the following mobility_model.graph

	parent --> child	sim(parent,child)
	1:1 --> 6:0			0.7733602811121824
	1:1 --> 2:0			0.7114914586232077
	2:1 --> 1:0			0.7114914586232077
	2:1 --> 6:0			0.39473684210526305
	3:1 --> 5:0			0.48719876576469
	3:1 --> 6:0			0.48472920895592575
	4:1 --> 5:0			0.4529108136578383
	4:1 --> 7:0			0.36927447293799814
	5:1 --> 3:0			0.48719876576469
	5:1 --> 4:0			0.4529108136578383
	6:1 --> 1:0			0.7733602811121824
	6:1 --> 3:0			0.48472920895592575
	7:1 --> 4:0			0.36927447293799814
	7:1 --> 3:0			0.17654696590094993
	8:1 --> 7:0			0.11941628680530642
	(only one other peer has similarity > 0.0 to peer 8)

	Graph generation procedure (see <self.generate_graph>):

	0. Calculate all pairwise similarities
	1. Split the peers into two disjoint subsets
	2. For all epochs t until time horizon T:
		2.1 Initialize nodes for every peer and epoch t
		2.2 Pick one of the peer_subsets alternatingly
		2.3 For every peer in peer_subset:
			2.3.1 sampleN users from the set of similar users
			2.3.2 generate an edge from the peer to every sampled other peer



	"""
	#################################
	# 0. RETURN FUNCTIONS           #
	#################################

	def sim_dict(self):
		""" Return the self.__sim_dict."""
		return self.__sim_dict

	def peers(self):
		""" Return the self.__peers."""
		return self.__peers

	#################################
	# 1. INITIALIZATION FUNCTIONS   #
	#################################

	def __init__(self, train_df, N, T, sim_string = "cosine", min_sim = 0.0, sim_dict = None, graph_seed = None, T_dict = None):
		# initialize peers and items first
		self.__train_df	= train_df
		self.__peers	= self.__train_df.loc[:,"userId"].astype(str).drop_duplicates().values
		self.__items	= self.__train_df.loc[:,"itemId"].astype(str).drop_duplicates().values
		# then call the template's initilization
		super().__init__(graph = DAG(peers = self.__peers), N = N, T = T, graph_seed = graph_seed, T_dict = T_dict)
		# mobility string for future reference
		self.mobility_string	= "DirectNMobility"
		self.sim_string 	= sim_string
		# create dictionaries that map the userIds and itemIds (given as <str> ids in the train_df
		# (coordinate format) to <int> indices of a 2-dim array (similarity matrix)
		self.__userId_dict	= {peer : i for i, peer in enumerate(self.__peers, 0)}
		self.__itemId_dict	= {item : i for i, item in enumerate(self.__items, 0)}
		self.sim_dict = sim_dict
		# customization parameters
		self.min_sim		= min_sim # minimum similarity only above which peers are considered as neighbors

		# initialize a random series of integers (generator object) that creates a unique series of integers
		self.random_gen         = random.Random(graph_seed)

		# (X) make it that either T or T_dict is None
		if (T_dict is not None):
			print("T_dict is not None. Setting T = None.")
			self.T = None


	#################################
	# 2. INITIALIZATION FUNCTIONS   #
	#################################
	# Calculate all pairwise distances

	def initialize_sim_dict(self):
		""" Initialize the similarity dictionary. """

		if self.__sim_dict is None:
			print("Initialize similarity dictionary via {} similarity.".format(self.sim_string))
			row  = self.__train_df.loc[:,"userId"].astype(str).values
			col  = self.__train_df.loc[:,"itemId"].astype(str).values

			# since row and col contain row and column identifiers of string type, we have to map them integer indices
			row  = [self.__userId_dict[userId] for userId in row]
			col  = [self.__itemId_dict[itemId] for itemId in col]

			# in order to row and col, we associate the ratings as np.float64
			data = self.__train_df.loc[:,"rating"].astype(np.float64).values

			# create scipy coordinate matrix
			D_coord		= coo_matrix((data, (row, col)))
			# cast to sparse
			D_sparse	= csr_matrix(D_coord)

			# (X) similarity measure selection
			if self.sim_string == "cosine":
				self.__sim_matrix = cosine_similarity(D_sparse, dense_output=True)
			if self.sim_string == "Pearson": # NOTE THAT THE REFERENCING IN DENSE DATAFRAMES IS DIFFERENT FROM SPARSE DATAFRAMES!
				# cast to array before applying pearson correlation (for it cannot handle sparse inputs)
				D_dense			= D_sparse.toarray()
				# n_jobs = -1 --> use all CPUs
				self.__sim_matrix = pairwise_distances(D_dense, metric=custom_pearsonr, n_jobs= -1)

			# get pairs of userIds of the 'upper triangle' ["userIds" are ordered as in self.__peers
			upper_triangle_indices = UpperTriangleGenerator(self.__peers)

			# intialize the sim_dict
			self.sim_dict = { userId1 + "-" + userId2 : self.__sim_matrix[self.__userId_dict[userId1], self.__userId_dict[userId2]]\
																					for userId1, userId2 in upper_triangle_indices}

		else:
			print("Use already initialized sim_dict for graph generation.")



	#########################################
	# 3. GENERATE GRAPH (MAIN FUNCTIONALITY)#
	#########################################

	def generate_graph(self):
		""" The generate graph method. """
		# 0. Calculate all pairwise similarities
		# initialize a dictionary of pairwise similarities
		# NOTE that there are no duplicates! That is the existence of a similarity on key "A-B"
		# (between some user "A" and "B" implies that the dictionary does not hold a similarity on key "B-A"
		# which would be identical
		self.initialize_sim_dict()

		# initialize the vertices at time t = 0 (for example ["A:0", "B:0"] for some peers ["A", "B"])
		self.graph.initialize_peers(t=0)


		# global time horizon
		#########################
		if self.T is not None:
			# 2. For all epochs t until time horizon T:
			for t in range(1,self.T+1,1):
				# 2.1 Initialize nodes for every peer and epoch t
				self.graph.initialize_peers(t=t)
				# initialize and fill peer_counter_df
				peer_counter_df = pd.DataFrame([self.N for _ in self.graph.peers()], index = self.graph.peers(), columns = ["counter"])
				# generate the graph at epoch <t>
				self.graph, peer_counter_df = self.generate_graph_at_t(self.graph, t, peer_counter_df, self.random_gen)

		# local time horizons
		########################
		else: # if self.T is None and self.T_dict is not None (see (X))
			# fill peer_counter_df
			max_time_horizon = max(self.T_dict.values())
			for t in range(1,max_time_horizon+1):
				# initialize peers at epoch t (although they might not have any connections)
				# If initialization is removed this causes an issue with the iteration over all epochs and peers in the DecAlgoTemplte.inner_for_loop
				# Notably 		children      = [el.split(":")[0] for el in self.graph.children(parent_at_t)]
				# which, it is not possible to fetch the children of a node that does not exist,
				# by initializing dummy nodes, an empty list of children is returned here.
				# NOTE that dummy nodes do not contribute to that node's local time horizon.
				self.graph.initialize_peers(t = t)
				# initialize peer_counter_df
				# CAVEAT here the peer_counter_df is expected to not hold 0 counts!
				non_zero_count_peers = [peer for peer in self.graph.peers() if t <= self.T_dict[peer]]
				peer_counter_df = pd.DataFrame([self.N for _ in non_zero_count_peers], index = non_zero_count_peers, columns = ["counter"])

				# generate the graph at epoch <t>
				self.graph, peer_counter_df = self.generate_graph_at_t(self.graph, t, peer_counter_df, self.random_gen)


	def generate_graph_at_t(self, graph, t, peer_counter_df, random_gen):
		peers = peer_counter_df.index
		# 1. Split the peers into two disjoint subsets
		first_peers  = peers[:len(peers)//2]
		second_peers = peers[len(peers)//2:]
		# 2.2 Pick one of the peer_subsets alternatingly
		peer_subset = first_peers if t%2 == 0 else second_peers
		# 2.3 For every peer in peer_subset:
		for peer in peer_subset:
			# 2.3.1 Fetch peers similar to peer and present in peers
			# NOTE that filtering is necessary as peers may be a subset of the peers, where similar peers are fetched from all peers present in <self.__sim_dict>.
			similar_peers = self.fetch_similar_peers(peer, peer_filter = peers)
			# 2.3.2 Check whether peer has any similar peers
			if len(similar_peers) > 0:
				# 2.3.3 sampleN similar users ############## replace self.N with the number in the peer_counter_df
				similar_Nsample_peers, random_gen = self.sampleN(random_gen, min(peer_counter_df.loc[peer,"counter"],len(similar_peers)), similar_peers)
				#similar_Nsample_peers, random_gen = self.sampleN(random_gen, min(self.N,len(similar_peers)), similar_peers)
				# if there are similar peers
				# 2.3.4 generate an edge from the peer to every sampled other peer
				for similar_sample_peer in similar_Nsample_peers:
					# link peer at time t to similar_sample_peer at time t-1
					source_key = peer+":"+str(t)
					target_key = similar_sample_peer+":"+str(t-1)
					graph.add_edge(source_key, target_key)
					# link peer at time t-1 to similar_sample_peer at time t
					source_key = similar_sample_peer+":"+str(t)
					target_key = peer+":"+str(t-1)
					graph.add_edge(source_key, target_key)

		return graph, peer_counter_df

	def fetch_similar_peers(self,peer, peer_filter = None):
		"""
			Reads all similar peers from the similarity dictionary self.__sim_dict.
			If <peer_subset> is not None, then only pick similar peers present in the peer_subset.
		"""
		# pick the similarities in which peer is involved (this should have len(self.__peers)-1
		peer_sim_dict = {key: [value] for key, value in zip(self.sim_dict.keys(), self.sim_dict.values()) if peer in key.split("-")}
		# find the N most similar peers to peer
		# <peer_sim_series> is now a pandas.Series
		peer_sim_series = pd.DataFrame.from_dict(peer_sim_dict).iloc[0,:]
		# drop all similarities that are not larger than self.min_sim; e.g. '5-1'
		similar_peer_pairs = peer_sim_series[peer_sim_series > self.min_sim].index
		# select the users that are not the user
		similar_peers = [peer_pair.split('-')[0] if peer_pair.split('-')[1] == peer else peer_pair.split('-')[1] for peer_pair in similar_peer_pairs]
		# filter similar peers that are present in peer_subset
		if peer_filter is not None:
			similar_peers = [el for el in similar_peers if el in peer_filter]

		return similar_peers

if __name__ == "__main__":
	# predefined dataset paths
	test_datapath        = os.path.join(root_dir, "./data/demo_df.csv")
	# load rating data as a pandas.DataFrame in coordinate format with the columns ["userId", "itemId", "rating"]
	train_df = load_ratings_df(test_datapath, userId_label = "userId", itemId_label = "movieId", rating_label = "rating", timestamp_label = "timestamp")#ml_smallest_datapath)
	N = 2   # number of connections per user and epoch on average (!)
	T = 4	# number of epochs; (height-1) of the resulting graph
	sim_string   	= "cosine"
	min_sim			= 0.0

	sim_dict		= {'1-2': 0.14, '1-3':0.612, '1-4':0.41, '1-5':0.895, '1-6':0.512, '1-7':0.671, '1-8':0.111, '2-3':0.4126, '2-4':0.162, '2-5':0.17823,\
	 					'2-6':0.81, '2-7':0.14, '2-8':0.1512, '3-4':0.124, '3-5':0.162, '3-6':0.125, '3-7':0.6712, '3-8':0.5029, '4-5':1.0, '4-6':0.95, \
						'4-7':0.9783, '4-8':0.123, '5-6':0.56978, '5-7':0.512, '5-8':0.761, '6-7':0.892,'6-8':0.521, '7-8':0.4123}

	graph_seed = 1234

	mobility_model = DirectNMobility(train_df, N, T, sim_string = sim_string, min_sim = min_sim, graph_seed = graph_seed)
	mobility_model.generate_graph()

	print("EDGES:",mobility_model.graph.edges())
	print("VERTICES:",mobility_model.graph.vertices())
	print("SIM_DICT",mobility_model.sim_dict())
