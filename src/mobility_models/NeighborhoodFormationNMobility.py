
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)
# for debugging
from utilities.util import load_ratings_df, make_train_test_split, build_execution_graph

# for production
from mobility_models.MobilityTemplate import MobilityTemplate
from mobility_models.graph import DAG
from utilities.util import UpperTriangleGenerator
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


class NeighborhoodFormationNMobility(MobilityTemplate):
	""" Class that defines a mobility model that assigns the peer's N most similar users.
	If there are less than M<N peers, then the peer will be assigned M neighbors. The model therefore
	requires pairwise similarities between the peers.

	NOTE1:	In this mobility model only one epoch (T=1) occurs in which every peer is matched with exactly its
			N most similar peers. It is therefore equivalent to traditional decentralized CF with neighborhood formation.

	NOTE2:	Similarity is expected to be symmetric. That is sim(a,b) = sim(b,a) for some peers a and b. It might be necessary to
			assume that the similarity is not symmetric, if for instance users hide parts of their profiles.

	Input:
		- train_df		: <pandas.DataFrame> a training dataframe to extract pairwise similarities from
						  train_df should have coordinate-format, that is include columns ["userId", "itemId", "rating"]
						  similarities will be calculated on the basis of user-profiles (that is ratings of a user)
		- N				: <int> neighborhood size, how many neighbors to assign to every peer.
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

	def __init__(self, train_df, N, sim_string = "cosine", min_sim = 0.0, sim_dict = None):
		# initialize peers and items first
		self.__train_df	= train_df
		self.__peers	= self.__train_df.loc[:,"userId"].astype(str).drop_duplicates().values
		self.__items	= self.__train_df.loc[:,"itemId"].astype(str).drop_duplicates().values
		# then call the template's initilization
		super().__init__(graph = DAG(peers = self.__peers), N = N, T = 1, graph_seed = None)
		# mobility string for future reference
		self.mobility_string	= "NeighborhoodFormationN"
		self.sim_string 	= sim_string
		# create dictionaries that map the userIds and itemIds (given as <str> ids in the train_df
		# (coordinate format) to <int> indices of a 2-dim array (similarity matrix)
		self.__userId_dict	= {peer : i for i, peer in enumerate(self.__peers, 0)}
		self.__itemId_dict	= {item : i for i, item in enumerate(self.__items, 0)}
		self.sim_dict = sim_dict
		# customization parameters
		self.min_sim		= min_sim # minimum similarity only above which peers are considered as neighbors

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



	#####################
	# 3. GENERATE GRAPH #
	#####################

	def generate_graph(self):
		""" The generate graph method. """
		# initialize a dictionary of pairwise similarities
		# NOTE that there are no duplicates! That is the existence of a similarity on key "A-B"
		# (between some user "A" and "B" implies that the dictionary does not hold a similarity on key "B-A"
		# which would be identical
		self.initialize_sim_dict()

		# initialize the vertices at time t = 0 (for example ["A:0", "B:0"] for some peers ["A", "B"])
		self.graph.initialize_peers()

		# initialize the vertices at time t = 1 (for example ["A:1", "B:1"] for some peers ["A", "B"])
		# NOTE that this initialization is meant to circumvent a error in DecAlgoTemplate.inner_for_loop for the case that
		# a peer has no neighbors (peers to whom he/she has positive similarity). It initialization of peers at t=1 is not done,
		# then the loop in DecAlgoTemplate.fill_snapshots will fail.
		# ERROR:
		#	File "/home/teichinger/sigir2020/src/algorithms/templates/DecAlgoTemplate.py", line 981, in inner_for_loop
    	#	children      = [el.split(":")[0] for el in self.graph.children(parent_at_t)]
  		#	File "/home/teichinger/sigir2020/src/mobility_models/graph.py", line 87, in children
    	#	return self.__graph_dict[vertex]
		self.graph.initialize_peers(t = 1)

		for peer in self.__peers:
			# pick the similarities in which peer is involved (this should have len(self.__peers)-1
			peer_sim_dict = {key: [value] for key, value in zip(self.sim_dict.keys(), self.sim_dict.values()) if peer in key.split("-")}
			# find the N most similar peers to peer
			# <peer_sim_series> is now a pandas.Series
			peer_sim_series = pd.DataFrame.from_dict(peer_sim_dict).iloc[0,:]
			# drop all similarities that are not larger than self.min_sim
			peer_sim_series = peer_sim_series[peer_sim_series > self.min_sim]
			# sort by similarities and pick the N top ones
			N_peer_sim_series = peer_sim_series.sort_values(ascending = False).iloc[:self.N]
			# choose their corresponding neighbor userIds [connections are of the format: "A-B" that indicate the similarity between "A" and "B"
			neighbor_connections = N_peer_sim_series.index.tolist()
			N_neighbors = [[el for el in connection.split("-") if el != peer][0] for connection in neighbor_connections]
			# NOTE that the neighbors have to be the children (e.g. vertex with key "neighbor:0" for some peer (parent) "B:1")
			# NOTE that the parent is the data receiver and the child the data sender
			# NOTE that the source_keys (cf. graph.py) are the parents and the target_keys are the children
			for neighbor in N_neighbors:
				# link peer at time t=1 with neighbor at time t=0
				source_key = peer+":1"
				target_key = neighbor+":0"
				self.graph.add_edge(source_key, target_key)



if __name__ == "__main__":
	# predefined dataset paths
	test_datapath        = os.path.join(root_dir, "./data/test")
	# load rating data as a pandas.DataFrame in coordinate format with the columns ["userId", "itemId", "rating"]
	training_df = load_ratings_df(test_datapath, userId_label = "userId", itemId_label = "movieId", rating_label = "rating", timestamp_label = "timestamp")#ml_smallest_datapath)
	N = 2               # size of the neighborhoods
	sim_string   	= "cosine"
	min_sim			= 0.0

	sim_dict		= {'1-2': 0.14, '1-3':0.612, '1-4':0.41, '1-5':0.895, '1-6':0.512, '1-7':0.671, '1-8':0.111, '2-3':0.4126, '2-4':0.162, '2-5':0.17823,\
	 					'2-6':0.81, '2-7':0.14, '2-8':0.1512, '3-4':0.124, '3-5':0.162, '3-6':0.125, '3-7':0.6712, '3-8':0.5029, '4-5':1.0, '4-6':0.95, \
						'4-7':0.9783, '4-8':0.123, '5-6':0.56978, '5-7':0.512, '5-8':0.761, '6-7':0.892,'6-8':0.521, '7-8':0.4123}

	mobility_model = NeighborhoodFormationNMobility(training_df, N, sim_string = sim_string, min_sim = min_sim, sim_dict = sim_dict)
	mobility_model.generate_graph()

	print("EDGES:",mobility_model.graph.edges())
	print("VERTICES:",mobility_model.graph.vertices())
	print("SIM_DICT",mobility_model.sim_dict())
