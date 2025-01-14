
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
import numpy as np
import random
import time
from itertools import product
import multiprocessing
from multiprocessing import Pool, Manager, Process

from mobility_models.graph import DAG
from mobility_models.MobilityTemplate import MobilityTemplate
from utilities.util import UpperTriangleGenerator, UpperTriangleGeneratorX, UpperTriangleGeneratorY, RandomBinaryIterator,  split_every_n


class TIJMobility(MobilityTemplate):
	""" Class that takes an empirical contact data set in tij format as input and converts it into an execution graph.

	CAVEAT Is is assumed that the TIJ dataset satisfies  i <= j, so every row corresponds to exactly one connection between i and j.


	"""
	#################################
	# 1. INITIALIZATION FUNCTIONS   #
	#################################

	def __init__(self, graph, path_to_tij_dataset, timestamp_aggregation):
		# call the template's initilization
		# NOTE that we set N=1 as a dummy entry, such that DecAlgoTemplate.fill_snapshots properly marks the peers that seek to establish a connection
		super().__init__(graph = graph, N = 1, T = None, graph_seed = None)
		# mobility string for future reference
		self.mobility_string	= "TIJMobility"
		self.timestamp_aggregation = timestamp_aggregation
		self.path_to_tij_dataset = path_to_tij_dataset
		# read tij dataset (pandas)
		# NOTE that the timestamps in self.tij_df correspond to epochs in DecAlgoTemplate
		self.tij_df = self.read_tij_dataset(path_to_tij_dataset, timestamp_aggregation)
		self.peers_to_ij_dict = None
		self.ijs_to_peer_dict = None
		self.make_peer_to_ij_dicts()

		# for compatibility with MobilityTemplate
		self.random_gen = None

	##########################################
	# 2. UTILITY FUNCTIONS                   #
	##########################################

	def floor_divide(self, x):
		return x//self.timestamp_aggregation +1

	def read_tij_dataset(self, path_to_tij_dataset, timestamp_aggregation, sep = " "):
		df = pd.read_csv(path_to_tij_dataset, names = ["timestamp", "i", "j"], sep = sep, dtype = {"t": 'Int64', "i":str, "j":str})

		min_timestamp = df.loc[:,"timestamp"].min() #
		max_timestamp = df.loc[:,"timestamp"].max()


		# subtract all timestamps by min_timestamp such that the new min_timestamp is zero
		df.loc[:,"timestamp"] = df.loc[:,"timestamp"] - min_timestamp
		# aggregate the timestamps, into intervals of <timestamp_aggregation> timestamps,
		# e.g.
		# 0,1,2,3..,99 (<timestamp_aggregation> = 100) => all mapped to timestamp 1
		# 100,101,..,199                               => mapped to timestamp 2
		df.loc[:,"timestamp"] = df.loc[:,"timestamp"].apply(self.floor_divide)#//timestamp_aggregation+1

		df = df.rename(columns = {"timestamp" : "epoch"})

		return df

	def make_peer_to_ij_dicts(self):

		unique_ij_s = pd.concat([self.tij_df.loc[:,"i"],self.tij_df.loc[:,"j"]], axis = 0).drop_duplicates().tolist()

		# CAVEAT: if self.graph.peers() and unique_ij_s do not have the same length, this might produce a key error
		# more specifically, any peer that is not in self.peers_to_ij_dict will not have any connections (see <self.generate_graph_at_t>)
		self.peers_to_ij_dict = {peer:ij for peer,ij in zip(self.graph.peers(), unique_ij_s) }
		self.ijs_to_peer_dict = {v: k for k, v in self.peers_to_ij_dict.items()}

		return

	def select_unique_ijs_in_tij_df(self, tij_df):
		"""
			Select the unique ijs in a tij_df (or subdataframe).
		"""
		return pd.concat([tij_df.loc[:,"i"], tij_df.loc[:,"j"]], axis = 0).drop_duplicates().tolist()

	def select_unique_connection_counterparts(self, peer_ij, tij_df):
		"""
			Select the counterparts (the target_ijs that some source_ij establishes a connection with) in some tij_df.

			e.g.
			Input:
				peer_ij: "a"
				tij_df :
					epoch	i	j
					1		"a"	"b"
					1		"b"	"c"
					1		"d"	"a"
			Output:
				["b", "d"]
		"""
		# select all counterparts
		list_of_all_counterparts = tij_df[(tij_df["i"] == peer_ij) | (tij_df["j"] == peer_ij)].loc[:,["i", "j"]].to_numpy().flatten().tolist()
		# drop duplicates
		list_of_unique_counterparts = list(dict.fromkeys(list_of_all_counterparts)) # drop duplicates
		# drop <peer_ij>; the if condition is to prevent an error in case the <peer_ij> is not in the <list_of_unique_counterparts>
		list_of_unique_counterpart_without_peer_ij = [unique_counterpart for unique_counterpart in list_of_unique_counterparts if unique_counterpart != peer_ij]

		return list_of_unique_counterparts

	##########################################
	# 3. GENERATE GRAPH [MAIN FUNCTIONALITY] #
	##########################################
	def generate_graph_at_t(self, graph, t, peer_counter_df, random_gen):
		""" Build the graph for epoch <t>.

		IGNORE <random_gen> since connections are already pre-defined.

		CAVEAT the timestamp in the tij dataset and the argument <t> of this function are not necessarily the same.


		"""
		mapped_peers_in_peer_counter_df = [peer for peer in self.peers_to_ij_dict.keys() if peer in peer_counter_df.index]
		# select the ijs that correspond to the peers that seek a connection in epoch t
		ijs_of_peer_counter_df = [self.peers_to_ij_dict[peer] for peer in mapped_peers_in_peer_counter_df]

		# select connections at epoch t
		tij_df_at_epoch_t = self.tij_df[self.tij_df["epoch"] == t]
		# select connections at epoch t between users in peer_counter_df; all connections correspond to connections in which only one or neither ij seeks to establish a connection
		tij_df_at_epoch_t_and_between_peers_in_tij_df = tij_df_at_epoch_t[tij_df_at_epoch_t["i"].isin(ijs_of_peer_counter_df) & tij_df_at_epoch_t["j"].isin(ijs_of_peer_counter_df)] #peer_counter_df.index)]  #[ (tij_df_at_epoch_t["i"] in peer_counter_df.index) | (tij_df_at_epoch_t["j"] in peer_counter_df.index)]
		# peer_counter_df.index is the list of peers who seek connections in epoch t
		for peer in mapped_peers_in_peer_counter_df:
			# look up the ij that corresponds to peer
			peer_ij = self.peers_to_ij_dict[peer]
			source_key = peer+":"+str(t)
			# CAVEAT HERE WE ASSUME THAT EVERY CONNECTION IS REPRESENTED BY ONLY ONE ROW IN THE TIJ DATASET (in other words, when "a" (in column "i") connects to "b" (in column "j"), then there is not combination "b" (in column "i") with "a" (in column "j"))
			# therefore, we not only need to add one edge from "i" to "j", but also the 'reverse' edge from "j" to "i"
			reverse_target_key = peer+":"+str(t-1)

			# look up the connections of <peer_ij> to <peer_ij_target> in <tij_df_at_epoch_t_and_between_peers_in_tij_df>
			unique_connection_counterparts_of_peer_ij = self.select_unique_connection_counterparts(peer_ij, tij_df_at_epoch_t_and_between_peers_in_tij_df)
			# add edges to all these counterparts
			for unique_connection_counterpart_of_peer_ij in unique_connection_counterparts_of_peer_ij:
				unique_connection_counterpart_of_peer = self.ijs_to_peer_dict[unique_connection_counterpart_of_peer_ij]
				target_key = unique_connection_counterpart_of_peer+":"+str(t-1)
				# add edge from <peer_ij> to <peer_ij_target> (the associated edge from <peer_ij_target> to <peer_ij> will be added when way will be added )
				graph.add_edge(source_key, target_key)
				# add the 'reverse' edge
				reverse_source_key = unique_connection_counterpart_of_peer+":"+str(t)
				# add the 'reverse' edge
				graph.add_edge(reverse_source_key, reverse_target_key)

		return graph, None


	def generate_graph(self):
		"""

		Build up a graph according to the connections defined in the <tij_df> dataset.

		"""
		# TODO:




if __name__ == "__main__":
	N 		= 1               # size of the neighborhoods
	T		= 2		  # number of epochs
	graph_seed	= 1234		  # for graph generation

	peers = ['A', 'B', 'C']

	graph			= DAG(peers=peers)
	# initialize the mobility model
	path_to_tij_dataset = "../../data/co-presence-data/tij_pres_SFHH.dat"

	#	def __init__(self, graph, path_to_tij_dataset , graph_seed = None):
	#		# call the template's initilization
	#		super().__init__(graph = graph, N = None, T = None, graph_seed = graph_seed)
	#		# mobility string for future reference
	#		self.mobility_string	= "TIJMobility"
	#		# read tij dataset (pandas)
	#		self.tij_df = self.read_tij_dataset(path_to_tij_dataset)
	timestamp_aggregation = 100
	mobility_model = TIJMobility(graph, tij_datapath, timestamp_aggregation)
	# generate the execution graph
	t=1
	peer_counter_df = pd.DataFrame(1, index = peers, columns=["counter"]) #, names = ["counter"])
	execution_graph = mobility_model.generate_graph_at_t(graph, t, peer_counter_df, None)

	print(execution_graph.edges())
	print(execution_graph.vertices())
