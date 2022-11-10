
# add upper directory to sys-path to find graph.py
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[1]) # ./src
root_dir   = str(file_dir.parents[2]) # ./
sys.path.insert(0, src_dir)


# for debugging
import os
from sklearn.model_selection import KFold
from mobility_models.graph import DAG
from mobility_models.AssignNMobility import AssignNMobility
from mobility_models.NeighborhoodFormationNMobility import NeighborhoodFormationNMobility
from utilities.util import df2Snapshot_refs, coord_cosine, coord_Pearson
import time

# for production
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import random
import pickle
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utilities.util import custom_pearsonr, UpperTriangleGenerator, get_current_timestamp, sampleN
from sklearn.metrics import pairwise_distances
from itertools import product, chain
from multiprocessing import Pool

# testing:
from multiprocessing import Queue


class DecAlgoTemplate(metaclass=ABCMeta):
	"""
	Parent class for DataMinRS simulation experiments as per (). It implements a basic framework for simulating decentralized user-based collaborative filtering.
	The class builds on a directed acyclic execution graph (DAG: see mobility_models.graph.DAG). The execution graph encodes contacts between users. This simulation
	simulates all encounters. Users are represented by (database) snapshots. The output of the simulation is a set of snapshots of all users at a given time.
	Every snapshot initially consists of the respective user's rating profile. The snapshots grow over time due to users exchanging rating data in contacts.

	Calling <self.fill_snapshots> starts the simulation. The function calculates all peers' (database) snapshots
	successively. The snapshots can be saved every i periods to the <self.output_dir>, where every snapshots will be saved to a distinct pandas DataFrame in .csv format
	for later evaluation.

	The <self.fill_snapshots> method in detail:

		I. Initialize simulation (snapshots, pairwise similarities ...)
		II. Iterate over all epochs t = 1,..,T and all peers
			II.1 For_loop (epoch t fixed) [can be parallelized over]
				II.1.1 Collect the payload [i.e. information to be shared by the children with the parent] over all children of a parent node
				II.1.2 Aggregate the information in the payload [if specified]
				II.1.3 Run a garbadge collection of the aggregated information in order to prevent for example duplicates
			II.2 [OPTIONAL] Update system parameters (theta, T (prune self.graph)); else use static parameters
			II.3 Merge the aggregated information with the currently available information in the snapshot (future_snapshots_dict[peer])
			II.4 Every save_every_i periods, save all snapshots to a designated output_dir, where every future snapshot is saved to a distinct .csv file
				NOTE that the train_df is only needed at training time

		After the snapshots have been saved to disk, the functions in evaluation.py can be used in order to evaluate the goodness of the algorithm

	Simulation classes inherit from this template class and only (!) implement the following abstract functions:
		* II.1.1 collect_payload
		* II.1.2 aggregate_payload
		* II.1.3 collect_garbadge

	REMARK(S):
		1. The terms 'user' and 'peer' are used interchangeably.
		2. The term 'initial' denotes that no contacts (and thus no data exchanges) have happened so far.
		3. The term 'future' denotes that contacts (and thus data data exchanges) have (!) happened so far.
		4. The term 'initial snapshot' denotes a user's rating profile
		5. Parallelization is only feasible within an epoch, and not across epochs, for what ratings can be exchanged in a contact depends on the previously collected ratings.

	Basic variables:

	- graph					: <mobility_models.graph.DAG> The execution graph that holds the pairwise encounters between peers.
	- train_df				: <pandas.DataFrame> training dataframe that holds the training ratings in coordinate format ["userId", "itemId", "rating"].
	 						  The initial snapshots (before any contact) are drawn from this dataframe.
	- peers					: <list> of all peers' identifiers specified in the train_df; [see column "userId"]
	- initial_snapshot_refs	: <dict> with userId keys and pandas.DataFrame.index values, used for fast user-profile lookups
	- future_snapshots_dict	: <dict> with userId keys and pandas.DataFrame values; every pandas.DataFrame represents the collected information of userId at time t
	- snapshot_info_string	: <str> specifying the parameter for this experiment (e.g. dataset, random_seeds for mobility or train/test split etc. (cf. <save_future_snapshots> for details)
	- dataset_string		: <str> identifying the underlying training dataset
	- output_dir			: <str> path of a directory to persist snapshots (one .csv file with name userId.csv per userId) that are persisted periodically (cf. self.__save_every_i)
	- sim_string			: <str> specifying the similarity measure to use for pairwise user similarity estimation
	- sim_dict				: <dict> for fast similarity lookups, format: { "peer1-peer2" : sim(peer1, peer2)}
	- userId_dict			: <dict> for referencing the userIds in the train_df with integers
	- itemId_dict			: <dict> for referencing the itemIds in the train_df with integers
	- sim_mat_path			: <str> path to a similarity matrix in coordinate format (: columns = [row (str; peer1), col(str; peer2), data(float)])

	- hide_p[deprecated]	: <float> between 0.0 and 1.0. Default None. When not None, hide <hide_p> of the raw profile data for similarity estimation. Hiding requires both hide_p and hide_seed.
	- hide_seed[deprecated]	: <int> random seed for the randomized hiding of raw profile entries for privacy-preserving similarity comparison. Hiding requires both hide_p and hide_seed.

	- dynamic_percentile	: <float> percentile of a peer's similarity histogram to use as threshold for sending data if above.
	- timestamp_delta		: <int> difference of epochs above which previously collected entries are not considered for inclusion into the payload. E.g. if timestamp_delta = 2 and t = 5
							  entries in the future_snapshots_dict with timestamp smaller than t-timestamp_delta = 5 - 2 = 3 are not considered for inclusion into te payload.
							  If <timestamp_delta> is None, all ratings are considered for inclusion into the payload in <self.collect_payload>.
							  NOTE that the entries in self.future_snapshots_dict are NOT dropped, yet filtered. Evaluation on the output snapshots need to filter as well.
							  Ratings are not dropped into order to be able to inspect all collected ratings.
	- anonymous_contacts	: [THIS HAS TO BE SPECIFIED, WHAT DOES THIS MEAN EXACTLY?, HOW ARE DUPLICATE CONTACTS DISCRIMINATED]
								<bool> if True, peers can identify previously met peers. In particular, payloads will not include multiple copies of
							  profiles of the same peer received from that peer in multiple contacts. if False, payloads will include multiple copies of the same
							  peer's profile if multiple contacts happened.	default: False

	- dynamic_percentile_dict: <dict> of the format {userId<str>:dynamic_percentile<float>}. If it is not specified, then it will be initialized with {userId:self.__dynamic_percentile} for all userIds (peers).


	[TO BE COMPLETED]
	Format:							  ALWAYS												ONLY if self.hide_p is not None
		self.__sim_dict:	keys	: [peer1+'-'+peer2]			 							OR	[peer1+'-'+peer2+':'+t]
							values	: similarity between initial_dfs of peer1 and peer2		similarity between peer1's initial_df and peer2's sampled profile
																							sampled profiles can be found in encounter_dict

		ONLY if self.hide_p is not None
		self.__encounter_dict:	keys	: [peer1+'-'+peer2+':'+t]
								values	: peer1's sampled profile for the edge (peer1:t, peer2:t-1) of the calculation graph (self.graph)



	"""

	def __init__(self, graph, train_df, snapshot_info_string = "", dataset_string = "", initial_snapshot_string= "", output_dir = os.path.join(root_dir,"data/snapshots"),\
						save_every_i = 5, sim_string = "cosine", sim_mat_path = None, min_sim_to_sender = 0.0, min_sim_to_child_child = 0.0, topN = 3,\
						min_sim_to_sender_dynamic = False, dynamic_percentile = None, time_horizon = None, hide_p = None, hide_seed = None, hide_parallel = True, sim_dict_pickle_string = None,\
						timestamp_delta = None, anonymous_contacts = False, processes = None, mean_centering = False, log_output_dir = os.path.join(root_dir,"data/parameter_logs"),\
						dynamic_percentile_dict = None, time_horizon_dict = None):
		self.graph	                 	= graph
		self.__train_df              	= train_df
		self.__peers				 	= None
		self.__items				 	= None
		self.__initial_snapshot_refs 	= None
		self.future_snapshots_dict		= None
		self.__processes				= processes
		self.mean_rating_dict			= None
		# for persisting snapshot information to disk
		self.__snapshot_info_string  	= snapshot_info_string
		self.__dataset_string		 	= dataset_string
		self.__output_dir            	= output_dir
		self.__save_every_i			 	= save_every_i
		self.algo_string			 	= "template"
		# for persisting parameter information to disk
		self.__log_output_dir			= log_output_dir
		# parameters for aggregation (including similarity estimation)
		self.sim_string            	= sim_string
		self.__sim_dict				 	= None
		self.__userId_dict			 	= None
		self.__itemId_dict			 	= None
		self.mean_centering			= mean_centering
		# technical parameter (global time ticker)
		self.t							= None
		self.__sim_mat_path				= sim_mat_path
		# for selection of data to be exchanged
		self.topN						= topN
		self.min_sim_to_sender	 		= min_sim_to_sender
		self.min_sim_to_child_child		= min_sim_to_child_child
		# DYNAMIC
		self.__min_sim_to_sender_dynamic= min_sim_to_sender_dynamic
		self.min_sim_to_sender_dict		= None
		self.min_sim_to_child_child_dict= None
		self.__dynamic_percentile		= dynamic_percentile
		self.dynamic_percentile_dict	= dynamic_percentile_dict
		self.__time_horizon				= time_horizon
		self.time_horizon_dict			= time_horizon_dict
		# for hiding profile data (and thus have non-constant pairwise similarities)
		self.__hide_seed				= hide_seed
		self.hide_p						= hide_p
		self.__encounter_dict			= None
		self.__hide_parallel			= hide_parallel
		# pickling of sim_dict
		self.sim_dict_pickle_dir		= os.path.join(root_dir,"data/similarity_dicts")
		self.initial_snapshot_refs_pickle_dir = os.path.join(root_dir,"data/snapshot_refs")
		self.__sim_dict_pickle_string	= str(sim_dict_pickle_string) # pickle sim_dict if not None
		self.__initial_snapshot_pickle_string = "snapshot_refs"+initial_snapshot_string # the pickle strings are identical for instance, only the output folders are different
		# garbadge collection, only for filtering in the <collect_payload> method instead of hardcoding deletion of old entries in <collect_garbadge>
		self.timestamp_delta			= timestamp_delta
		# anonymity
		self.anonymous_contacts			= anonymous_contacts
		# parameter_control
		self.parameter_control_model	= None

	#####################################################
	# 1. FUNCTIONS FOR SNAPSHOT MANAGEMENT              #
	#####################################################

	def default_snapshot_row(self):
		""" Define the default snapshot row as an empty pandas DataFrame that holds the column
		identifiers for the future_snapshots_dfs. """
		return pd.DataFrame([], columns = self.default_snapshot_columns())

	def default_snapshot_columns(self):
		""" Define the default snapshot columns for the default snapshot row.

		- userId		: the tenant of the row
		- itemId		: the item identifier the row is on
		- rating		: rating/aggregated rating [ratings in self.__train_df						denote "userId"'s rating,
											ratings in self.future_snapshots_dict[userId] denote a (aggregated) rating that "userId" received, yet is most likely to be by another user
		- sim			: the similarity of the child_child to the parent which is needed for the calculation of the aggregated similarity

						  - in <self.collect_payload> it will be given
						  - in <self.aggregate_payload> it will be used for agg_sim and then set to NaN
						  - sim is NaN in self.future_snapshots_dict since self.future_snapshots_dict is the knowledge that every peer has which
							does not include the similarity bound within the aggregated_similarity
		- agg_sim		: aggregated similarity
		- sender		: "userId" of the child (sender) that sends rating information to the parent (receiver)
		- sim_to_sender	: similarity of the child (sender) to the parent (receiver), this similarity value used for the calculation of the aggregated similarity

		"""
		return ["userId", "itemId", "rating", "sim", "agg_sim", "sender", "sim_to_sender", "timestamp"]

	def default_snapshot_dtypes(self):
		""" Define the default snapshot dtypes. needs to conform to <DecAlgoTemplate.default_snapshot_columns>.

		"""
		#       userId, itemId, rating, sim,   agg_sim,  sender,  sim_to_sender,   timestamp
		return [str,    str,     float, float, float,    str,     float,           int]

	def add_foldNr(self, foldNr):
		""" Add the foldNr to the snapshot_info_string. """
		self.__snapshot_info_string += "_foldNr={}".format(str(foldNr))

	def save_parameters(self):
		""" Save <self.dynamic_percentile_dict> and the timehorizons per user in the self.algorithm.graph> execution graph.
			Output:

		"""

		# if the output directory does not exist, create the directory
		if not os.path.isdir(self.__log_output_dir):
			os.makedirs(self.__log_output_dir)

		log_output_path = os.path.join(self.__log_output_dir, os.path.split(self.__output_dir)[-1] + self.algo_string + self.__snapshot_info_string + ".log")

		# reset the parameter log file at the start of the experiment (t==1)
		if self.t == 1:
			with open(log_output_path, mode = "w") as f:
				# log column names
				f.write("userId, theta, T\n".format(self.t))


		# append a row for every peer; format: peer, param_value
		with open(log_output_path, mode = "a") as f:
			# log epoch <t>
			f.write("t={}\n".format(self.t))

			# log parameter values per userId
			for peer in self.__peers:
				f.write(peer+', '+str(self.dynamic_percentile_dict[peer])+ ', ' + str(self.time_horizon_dict[peer]) + '\n')

	def save_future_snapshots(self, verbose = True, early_stop = False):
		""" Save self.future_snapshots_dict entries to distinct csv files in an output_directory.

		Input:
			- t                    : <int> of the current snapshot's time
			- verbose              : <bool> if True: Print information on where the snapshots have been persisted
			- snapshot_info_string : <str> holding information on the experiment to add to the output directory

		Output:
			Create an output folder in the self.__output_dir that is named after the snapshot_info_string plus time t-1
			In that folder, for every peer, put a .csv file with the name <peer>.csv that holds the peer's future snapshot for
			evaluation. """
		current_time_string = get_current_timestamp()
		output_name = current_time_string
		# if t given, add time info
		if self.t != None:
			if early_stop:
				output_name += "_t={}[stop]".format(str(self.t))
			else:
				output_name += "_t={}".format(str(self.t))
		# if snapshot_info_string is given also add it
		if self.__snapshot_info_string != "":
			output_name += self.__snapshot_info_string
		output_name += "_" + self.algo_string
		# assemble the individual output_path
		output_path = os.path.join(self.__output_dir, output_name)
		# if the output directory does not exist, create the directory
		if not os.path.isdir(output_path):
			os.makedirs(output_path)

		# now for all peers create a .csv file with the name <peer>.csv in <output_path>
		for peer in self.future_snapshots_dict.keys():
			future_snapshot = self.future_snapshots_dict[peer]
			# write the .csv file
			future_snapshot.to_csv(os.path.join(output_path, peer + ".csv"))

		if verbose:
			print("Saved snapshots at {}".format(output_path))

	def save_sim_dict(self):
		""" Persist the similarity information in the sim dict to a csv file in coordinate format at self.__output_dir. """
		current_time_string = get_current_timestamp()
		with open(os.path.join(self.__output_dir, current_time_string + "_sim_mat"), mode = "w") as f:
			for key in self.__sim_dict.keys():
				# only save the pairwise similarities of non-hidden (subsampled) user profiles
				if ":" not in key:
					peer1, peer2 = key.split("-")
					sim   = self.__sim_dict[key]
					line = peer1+","+peer2+","+str(sim)+"\n"
					f.write(line)

	#####################################################
	# 2. GET FUNCTIONS                                  #
	#####################################################

	def use_parameter_control_model(self, parameter_control_model):
		self.parameter_control_model = parameter_control_model
		return

	def get_graph(self):
		""" Return the graph . """
		return self.graph

	def set_graph(self, new_graph):
		self.graph = new_graph

	def get_dynamic_percentile_dict(self):
		""" Return the dynamic_percentile_dict. """
		return self.dynamic_percentile_dict

	def set_dynamic_percentile_dict(self, new_dynamic_percentile_dict):
		self.dynamic_percentile_dict = new_dynamic_percentile_dict
		return

	def get_time_horizon_dict(self):
		""" Return the time_horizon_dict. """
		return self.time_horizon_dict

	def set_time_horizon_dict(self, new_time_horizon_dict):
		self.time_horizon_dict = new_time_horizon_dict
		return

	def initial_snapshots(self):
		""" Return the initial snapshots. """
		return self.__initial_snapshots

	def train_df(self):
		""" Return the train_df. """
		return self.__train_df

	def val_df(self):
		""" Return the val_df. """
		return self.__val_df

	def peers(self):
		""" Return the peers. """
		return self.__peers

	def initial_snapshot_refs(self):
		""" Return the initial snapshot refs. """
		return self.__initial_snapshot_refs

	def get_future_snapshots_dict(self):
		""" Return the future_snapshots_dict. """
		return self.future_snapshots_dict

	def snapshot_info_string(self):
		""" Return the snapshot_info_string. """
		return self.__snapshot_info_string

	def dataset_string(self):
		""" Return the dataset_string. """
		return self.__dataset_string

	def output_dir(self):
		""" Return the output_dir. """
		return self.__output_dir

	def save_every_i(self):
		""" Return save_every_i. """
		return self.__save_every_i

	def mean_centering(self):
		""" Return mean_centering. """
		return self.mean_centering

	def sim_string(self):
		""" Return sim_string. """
		self.sim_string

	def sim_dict(self):
		""" Return sim_dict (with pairwise similarities between peers """
		return self.__sim_dict

	def userId_dict(self):
		""" Return the userId_dict. """
		return self.__userId_dict

	def itemId_dict(self):
		""" Return the itemId_dict. """
		return self.__itemId_dict

	def hide_seed(self):
		""" Return the hide_seed value. """
		return self.__hide_seed

	def hide_p(self):
		""" Return the hide_p value. """
		return self.hide_p

	def timestamp_delta(self):
		""" Return the timestamp_delta value. """
		return self.timestamp_delta

	def anonymous_contacts(self):
		""" Return the anonymous_contacts value. """
		return self.anonymous_contacts

	#####################################################
	# 3. UTILITY FUNCTIONS                              #
	#####################################################

	def center_mean(self, profile, per_userId = False):
		""" For a given profile <pandas.DataFrame> .

		Input:
			- profile:		<pandas.DataFrame> with at least a "rating" column
			- per_userId:	<bool> if True: subtract mean values grouped by entries in the "userId" column; DEFAULT: False

		Output:
			- profile: where the ratings in the "rating" column are subtracted by the average of the "rating" column
		"""
		if per_userId:
			if self.mean_rating_dict is None:
				self.initialize_mean_rating_dict()
			# subtract the per userId mean rating
			profile.loc[:,"rating"] = profile.apply(lambda x: x["rating"]- self.mean_rating_dict[x["userId"]], axis = 1)
		else:
			profile.loc[:,"rating"] = profile.loc[:,"rating"] - profile.loc[:,"rating"].mean()

		return profile

	def get_percentile_sim(self, peer):
		""" For a userId, calculate via the percentile similarity of that peer to all other peers.
		E.g. self.__dynamic_percentile = 0.9 --> get the treshold below which the 90% least pairwise similar peers are.
		"""
		if self.__sim_dict is None:
			self.initialize_sim_dict()
		if self.dynamic_percentile_dict is None:
			self.initialize_dynamic_percentile_dict()


		peer_sims = [self.fetch_similarity(peer, other_peer) for other_peer in self.__peers if other_peer != peer]
		return np.percentile(peer_sims, q= self.dynamic_percentile_dict[peer]*100)

	def get_percentile_sim_global(self):
		""" Calculate the global percentile (self.__dynamic_percentile) of all pairwise similarities via self.sim_dict.
		E.g. self.__dynamic_percentile = 0.9 --> get the similarity treshold below which the 90% of all pairwise similarities are.
		"""
		if self.__sim_dict is None:
			self.initialize_sim_dict()
		if self.dynamic_percentile_dict is None:
			self.initialize_dynamic_percentile_dict()

		peer_sims = list(self.__sim_dict.values()) # [self.fetch_similarity(peer, other_peer) for other_peer in self.__peers if other_peer != peer]
		return np.percentile(peer_sims, q = self.__dynamic_percentile*100)


	def fetch_similarity(self, peer1, peer2, t = None):
		""" Fetch a precalculated similarity between some peer1 and peer2 of self.__peers.
		Recall that self.create_encounterkey_sim_pairs foresees that peer1 is the similarity assessor.

		NOTE	that this function requires <initialize_sim_dict> to be run beforehand.
		NOTE	if t is not None, then peer1 assesses the similarity between peer1 and peer2. This is may be relevant
				in case peer1 and peer2 apply for instance distinct similarity measures.

		Input:
			- peer1	: <str> identifier of a peer 		[similarity assessor]
			- peer2	: <str> identifier of another peer
			- t		: <str> epoch (optional; useful if for instance profiles change over time or profiles shared are not entirely shared [see self.hide_p is not None])
		"""
		# if both peers are identical, trivially the similarity is maximal (=1.0)
		if peer1 == peer2:
			return 1.0

		# if t is not None, then there are always two complements of encounterkeys "1-2:3" AND "2-1:3"
		if t is not None:
			return self.__sim_dict[peer1+"-"+peer2+":"+t]

		else:
			# since the sim_dict only contains the 'upper triangle' of the similarity matrix, we
			# have to check the dictionary key "peer1-peer2" first, if it fails to return a value
			# we know that "peer2-peer1" exits, and vice versa.
			try:
				return self.__sim_dict[peer1+"-"+peer2]
			except:
				return self.__sim_dict[peer2+"-"+peer1]

	def flatten(self, list_of_lists):
		""" Quickly flatten a list of lists. """
		return list(chain.from_iterable(list_of_lists))

	def fetch_sub_train_df(self, peer1, peer2 = None, t = None):
		""" Fetch the sub-dataframe of self.__train_df that has userId == peer. Create a sub_df distinct from self.__train_df.
			Assignments to the returned sub-dataframe will thus not affect self.__train_df.

			NOTE that if changes to elements in self.__train_df are desired, self.__train_df needs to be indexed directly
			in order to avoid chained indexing.

		Input:
			- peer1	: <str>
			- peer2 : <str>
			- t		: <str>
		Output:
			- 		: <pandas.DataFrame> sub_frame of self.__train_df
		"""
		if (peer2 is not None) & (t is not None):
			peer_indices = self.__initial_snapshot_refs[peer1+"-"+peer2+":"+t]
		else:
			peer_indices = self.__initial_snapshot_refs[peer1]

		sub_train_df = self.__train_df.loc[peer_indices, :].copy()

		return sub_train_df

	def fast_concat(self, list_of_pandas_df):
		""" Fast concatenation as a single function of a list of pandas DataFrames via numpy.concatenate """
		# filter out empty dataframes
		list_of_pandas_df = [df for df in list_of_pandas_df if df.size != 0]
		# cast dataframes to type numpy.array
		list_of_numpy_df  = [df.to_numpy() for df in list_of_pandas_df]

		# concatenate numpy.arrays vertically (axis=0) with numpy.concatenate
		if list_of_numpy_df == []:
			result_df = self.default_snapshot_row()
		else:
			result_df = pd.DataFrame(np.concatenate(list_of_numpy_df, axis = 0), columns = self.default_snapshot_columns())

		# set data types
		# default_snapshot_columns(self):
		#col_type_dict = {col:type_str  for col, type_str in zip(self.default_snapshot_columns(), self.default_snapshot_dtypes())}
		#print(result_df)
		#result_df = result_df.astype(col_type_dict)
		#print(result_df)

		return result_df

	def map_userIds(self):
		""" Map all userIds (ordered) from self.__peers and assign them integer indices for the similarity matrix.

		Output:
			- self.__userId_dict : <dict> of the format {userId: index in similarity_matrix}
		"""
		if self.__peers is None:
			self.initialize_peers()

		self.__userId_dict = {peer : i for i, peer in enumerate(self.__peers,0)}

	def map_itemIds(self):
		""" Map all itemIds (ordered) from self.__peers and assign them integer indices for the similarity matrix.

		Output:
			- self.__itemId_dict : <dict> of the format {userId: index in similarity_matrix}
		"""
		if self.__items is None:
			self.initialize_items()

		self.__itemId_dict = {item : i for i, item in enumerate(self.__items,0)}

	def contactedge2encounterkey(self, contact_edge):
		""" Tranform a contact_edge from the execution graph in the format ("1:4", "2:3") into an encounter key "1-2-4" ("1" meets "2" at time t=4). """
		peer1 = contact_edge[0].split(":")[0]# ("1:4", "2:3") --> "1"
		peer2 = contact_edge[1].split(":")[0]# ("1:4", "2:3") --> "2"
		t     = contact_edge[0].split(":")[1]# ("1:4", "2:3") --> "4"
		encounter_key = peer1 + "-"+peer2+":"+t
		return encounter_key

	def create_encounterkey_sim_pairs_inparallel(self):
		""" For all encounterkeys, calculate the similarity used for aggregation in parallel.

		Output:
			- encounterkey_sim_pairs_all
		"""
		encounterkey_sim_pairs = []
		# start worker pool
		p_hide_sim_loop = Pool()
		results_object = p_hide_sim_loop.map_async(self.create_encounterkey_sim_pairs, iter(self.__encounter_dict.keys()), callback = encounterkey_sim_pairs.append)
		# wait until all results have been calculated
		results_object.wait()

		# encounterkey_sim_pairs is now a list of  (NOTE that map_async preserves the order of the peers)
		# single flatten necessary, the result will be
		encounterkey_sim_pairs = self.flatten(encounterkey_sim_pairs)

		# close and join the pool workers
		p_hide_sim_loop.close()
		p_hide_sim_loop.join()
		return encounterkey_sim_pairs

	def create_encounterkey_sim_pairs(self, encounterkey):
		""" For a given encounterkey, calculate the similarity used for aggregation.

		NOTE	that similarity sim(peer1, peer2) != sim(peer2, peer1) in all generality. Consider for instance the case that self.hide_p is not None.
				Then every peer only presents a portion of their ratings to the other peer!

				If however, self.hide_p is None, then sim(peer1, peer2) = sim(peer2, peer1) always.

		Format encounterkey:
				"1-2:3"; peer "1" meets peer "2" at epoch "3".
							OR EQUIVALENTLY
				"peer1-peer2:t" <--> "similaritycalculator-counterpart:t"

		Input:
			- encounterkey	:	<str> of the above format

		Output:
			- encounterkey	:	equivalent to the input
			- sim			:	similarity [assessed by peer1; who potentially receives only a portion of peer2's profile]

		"""
		peer1 = encounterkey.split("-")[0]
		peer2 = encounterkey.split("-")[1].split(":")[0]
		str_t = encounterkey.split(":")[1]

		df1 = self.fetch_sub_train_df(peer1)
		df2 = self.fetch_sub_train_df(peer2, peer1, str_t)
		if self.sim_string == "cosine":
			sim = coord_cosine(df1, df2, adjust_mean = False, aggregated = False)
		if self.sim_string == "Pearson":
			sim = coord_Pearson(df1, df2, adjust_mean = False, aggregated = False)

		return (encounterkey, sim)

	def create_encounterkey_userId_sub_df_pairs_inparallel(self):
		""" PARALLEL VERION: For a given userId, create pairs of encounterkey (format: ) and userId_sub_df (as a subdf of the userIds entries in self.__train_df) that have been subsampled
			to hold 1-self.hide_p of the original entries. For sampling use the global self.__hide_seed.
		Input:
			- userId
		Output:
			- encounterkey_userId_sub_df_pairs	: <tuple> duple of encounterkey (<str> format: ) and userId_sub_df (<pandas.DataFrame> subframe of self.__train_df)

		NOTE that this function cannot be (trivially) parallelized since the random generator requires the outputs of the previous roll (sampling result).
		"""
		encounterkey_userId_sub_df_pairs = []

		p_hide_loop = Pool()

		results_object = p_hide_loop.map_async(self.create_encounterkey_userId_sub_df_pairs, self.__peers, callback = encounterkey_userId_sub_df_pairs.append)
		# wait until all results have been calculated
		results_object.wait()

		# encounterkey_userId_sub_df_pairs_all is now a list of  (NOTE that map_async preserves the order of the peers)
		# double flatten necessary, the result will be a list of duples (encounterkey, userId_sub_df_pairs_all)
		encounterkey_userId_sub_df_pairs = self.flatten(encounterkey_userId_sub_df_pairs)
		encounterkey_userId_sub_df_pairs = self.flatten(encounterkey_userId_sub_df_pairs)

		# close and join the pool workers
		p_hide_loop.close()
		p_hide_loop.join()

		return encounterkey_userId_sub_df_pairs



	def create_encounterkey_userId_sub_df_pairs(self, userId, refs = False):
		""" For a given userId, create pairs of encounterkey (format: ) and userId_sub_df (as a subdf of the userIds entries in self.__train_df) that have been subsampled
			to hold 1-self.hide_p of the original entries. For sampling use the global self.__hide_seed.
		Input:
			- userId	: <str> identifier of the peer
			- refs 		: <bool> if True, instead of the sub_df (pandas.DataFrame) save the references to the self.__train_df only (as row indices)
		Output:
			- encounterkey_userId_sub_df_pairs	: <tuple> duple of encounterkey (<str> format: ) and userId_sub_df (<pandas.DataFrame> subframe of self.__train_df)

		NOTE that this function cannot be (trivially) parallelized since the random generator requires the outputs of the previous roll (sampling result).
		"""
		if (self.__hide_seed is None) & (self.hide_p is not None):
			print("CAVEAT hide_seed is not set, the results will not be reproducible!")

		random_gen  = random.Random(self.__hide_seed)
		encounterkey_userId_sub_df_pairs = []
		# get the number of contacts of userId in the execution graph (self.graph)
		# NOTE that the edges are of the form:
		#[('1:1', '2:0'), ('1:1', '3:0'), ('1:1', '5:0'), ('2:1', '1:0'), ('3:1', '1:0')]
		# it suffices to look for those edges (duples) that show the userId in the left part of the first string
		# e.g. if userId == "1", we are interested in the edges ('1:1', '2:0'), ('1:1', '3:0'), ('1:1', '5:0')
		contact_edges_of_userId = [el for el in self.graph.edges() if el[0].split(":")[0] == userId]
		# transform the contact_edges into encounters; ("1:4", "2:3") --> "1-2-4" ("1" meets "2" at time t=4)
		encounterkeys_of_userId = [self.contactedge2encounterkey(el) for el in contact_edges_of_userId]
		# pick the initial_snapshot of userId
		userId_df = self.fetch_sub_train_df(userId)
		# select the groundtruth_list as the indices of the initial_snapshot_df to sample N elements from
		groundtruth_list = userId_df.index.tolist()
		N = int(len(groundtruth_list)*(1-self.hide_p))
		# for all encounters of userId sample self.hide_p of the initial_snapshot_df (profile information provided in self.__train_df)
		# and store it in the encounter dictionary with keys encounterkey and values userId_sub_df
		##########################
		# FORMAT encounterkey:
		# "peer1-peer2:t" <--> "sender-recipient:t"
		##########################
		for encounterkey in encounterkeys_of_userId:
			sub_indices, random_gen, groundtruth_list = sampleN(random_gen, N, groundtruth_list, must_sample_list = None, drop_duplicates = False)
			# if refs is True: save the references (as row indices to self.__train_df instead of (shallow copies of the self.__train_df)
			if refs:
				# cast to pandas.Index, this type conversion is not necessary though.
				sub_indices = pd.Index(sub_indices)
				encounterkey_userId_sub_df_pairs.append((encounterkey, sub_indices))
			else:
				userId_sub_df = userId_df.loc[sub_indices, :]
				encounterkey_userId_sub_df_pairs.append((encounterkey, userId_sub_df))
		return encounterkey_userId_sub_df_pairs



	#####################################################
	# 4. INITIALIZATION FUNCTIONS                       #
	#####################################################

	def initialize_experiment(self):
		""" Initialize an experiment in order to run <self.fill_snapshots> and thus run the simulation. """
		# 16 April 2022
		#self.initialize_graph_T()

		self.initialize_peers()

		self.initialize_items()

		self.initialize_future_snapshots_dict()

		self.initialize_initial_snapshot_refs()

		self.initialize_sim_dict()

		self.initialize_dynamic_percentile_dict()

		self.initialize_time_horizon_dict()

		self.initialize_min_sim_to_sender_dict()

		if self.mean_centering:
			self.initialize_mean_rating_dict()

		if self.parameter_control_model.val_df is not None:
			self.parameter_control_model.initialize_val_dfs()

	def initialize_mean_rating_dict(self):
		"""
			Initialize the self.mean_rating_dict as a dictionary with <userId> keys and mean rating of <userIds> in the self.__train_df.
		"""
		self.mean_rating_dict = self.__train_df.groupby("userId").agg({"rating":"mean"}).to_dict()["rating"] # example: {<userId1>: 3.0, <userId2>: 3.5}
		return

	def initialize_peers(self):
		""" Initialize the list of peers to use for pairwise data exchanges.
			NOTE these peers might not match all the peers in self.graph, since the graph has been initialized with
				 the entire unique userIds in rating_df. The probability is very high, though, that they are identical. """
		if self.__peers is None:
			self.__peers = self.__train_df.loc[:,"userId"].astype(str).drop_duplicates().tolist() # TRY: list(dict.fromkeys(child_children)); perhaps faster

	def initialize_items(self):
		""" Initialize the list of itemIds. """
		if self.__items is None:
			self.__items = self.__train_df.loc[:,"itemId"].astype(str).drop_duplicates()

	def initialize_future_snapshots_dict(self):
		""" Procedure to fill empty pandas dataframes of a given format self.__default_snapshot_columns() to the keys in the dictionary in order not to have
		lookup errors in e.g. collect_payload.
		NOTE Requires self.__peers to be initialized
		"""
		# future_snapshots_dict requires self.__peers
		if self.__peers is None:
			self.initialize_peers()

		if self.future_snapshots_dict is None:
			self.future_snapshots_dict = dict()
			for peer in self.__peers:
				self.future_snapshots_dict[peer] = self.default_snapshot_row()

	def initialize_initial_snapshot_refs(self):
		""" Initialize the initial_snapshots_refs.
		NOTE Requires self.__peers to be initialized
		"""
		if self.__peers is None:
			self.initialize_peers()

		if self.__initial_snapshot_refs is None:
			self.__initial_snapshot_refs = { peer : self.__train_df[self.__train_df["userId"] == peer].index for peer in self.__peers}

	def initialize_sim_dict(self):
		""" Initialize a similarity matrix with pairwise similarities precalculated for future reference.
			If sim_string == "cosine", the output is a sparse matrix (scipy.sparse.scr) and has another referencing scheme
			than the dense outputs from e.g. "Pearson".

		Output:
			- self.__sim_dict: <dict> with keys [peer1+"-"+peer2] and values : similarity between peer1 and peer2
								if hide_p is not None, then calculate distinct subsamples of the original (initial_snapshot_refs) profiles for similarity comparison
								where hide_p determines the percentage of entries to drop
								Then the dictionary is extended with additional similarity values with
								keys [peer1+"-"+peer+":"+t]


		NOTE: When self.hide_p is not None, the sim_dict is enlarged with encounter similarities that are based on subsamples of the peers' train_df profile information
			  This introduces variance to the similarity estimation! This procedure follows Duriakova et al. (2019) in their 'Layer 1 Privacy'
		"""
		# check whether a custom sim_dict has been specified
		# NOTE that the DecAlgoTemplate class is agnostic of for instance the traintest_seed, and the mobility_seed used to split the data into train-test sets and create the execution graph respectively.
		#		We thus need to trust that the self.__sim_dict_pickle_string holds all necessary similarity information
		sim_dict_pickle_path				= os.path.join(self.sim_dict_pickle_dir, self.__sim_dict_pickle_string)
		initial_snapshot_refs_pickle_path	= os.path.join(self.initial_snapshot_refs_pickle_dir, self.__initial_snapshot_pickle_string)

		# create the output dir for the pickled sim_dict, if it does not exist already
		if not os.path.isdir(self.sim_dict_pickle_dir):
			os.mkdir(self.sim_dict_pickle_dir)
		# create the output dir for the initial snapshtos refs, if it does not exist already
		if not os.path.isdir(self.initial_snapshot_refs_pickle_dir):
			os.mkdir(self.initial_snapshot_refs_pickle_dir)


		# check whether a prepickled sim_dict exists
		if os.path.isfile(sim_dict_pickle_path):
			print("Found prepickled sim_dict file! Load sim_dict at {} instead of recalculation.".format(sim_dict_pickle_path))
			with open(sim_dict_pickle_path, mode = "rb") as f:
				self.__sim_dict = pickle.load(f)
			# set self.__sim_dict_pickle_string to None in order to avoid repickling
			self.__sim_dict_pickle_string = None


		# check whether a prepickled initial_snapshot_refs exists
		if os.path.isfile(initial_snapshot_refs_pickle_path):
			print("Found prepickled initial_snapshot_refs file! Load snapshot_refs at {} instead of recalculation.".format(initial_snapshot_refs_pickle_path))
			with open(initial_snapshot_refs_pickle_path, mode = "rb") as f:
				self.__initial_snapshot_refs = pickle.load(f)
			# set self.__initial_snapshot_pickle_string to None in order to avoid repickling
			self.__initial_snapshot_pickle_string = None

		# if self.__sim_dict could not be loaded from a prepickled sim_dict file,
		# initialize it with similarities specified by the self.sim_string (e.g. "cosine" for cosine similarity)
		if self.__sim_dict is None:
			# initialize an empty sim_dict
			self.__sim_dict = dict()
			# if not precalculated sim_mat at self.__sim_math_path can be found
			if self.__sim_mat_path is None:
				# initialize index map of userIds; self.__userId_dict is now usable
				self.map_userIds()
				self.map_itemIds()

				#
				# TODO: add "from_precalculated" option

				# Calculate all pairwise distances
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
				self.__sim_dict = { userId1 + "-" + userId2 : self.__sim_matrix[self.__userId_dict[userId1], self.__userId_dict[userId2]]\
																							for userId1, userId2 in upper_triangle_indices}
			# else use the self.__sim_mat_path to load a precalculate similarity matrix
			else:
				# if the matrix is in coordinate format
				coo_similarity_matrix = pd.read_csv(self.__sim_mat_path, header = None, names = ["peer1", "peer2", "sim"])
				for i in coo_similarity_matrix.index:
					peer1	= coo_similarity_matrix.loc[i,"peer1"].astype(str)
					peer2	= coo_similarity_matrix.loc[i,"peer2"].astype(str)
					sim 	= coo_similarity_matrix.loc[i,"sim"  ]

					self.__sim_dict[peer1+"-"+peer2] = sim

		"""
		# if self.hide_p is not None, add also hiddenprofiles (for additional privacy) for every peer and transaction (data exchange) cf. doc_string
		if (self.hide_p is not None) & (self.__sim_dict_pickle_string is not None):
			# initialize the encounter_dict
			#self.__encounter_dict = dict()
			########################
			# FOR TESTING
			########################
			self.__hide_parallel = False
			# Calculate the encounterkey_userId_sub_df_pairs_all (for the creation of the self.__encounter_dict)
			######################
			if self.__hide_parallel:
				# parallelly calculate the encounterkey_userId_sub_df_pairs
				hidden_sub_df_pairs_all = self.create_encounterkey_userId_sub_df_pairs_inparallel()

			else:
				# sequentially calculate the encounterkey_userId_sub_df_pairs
				hidden_userId_sub_df_ref_pairs_all = []
				for userId in self.__peers:
					hidden_userId_sub_df_ref_pairs = self.create_encounterkey_userId_sub_df_pairs(userId, refs = True)
					hidden_userId_sub_df_ref_pairs_all += hidden_userId_sub_df_ref_pairs
			# assign all encoutnerkey_userId_sub_df_pairs to key-value pairs in self.__encounter_dict
			for encounterkey, sub_df_ref in hidden_userId_sub_df_ref_pairs_all:
				self.__initial_snapshot_refs[encounterkey] = sub_df_ref

			# Calculate the similarities for the encounters on the basis of the encounter_dict
			################################################################################################
			# Now for all encounters (that are represented by 2 edges in the execution graph each, calculate the similarities between their respective sampled profiles
			# e.g. "1-2-4", "1" meets "2" at t=4; Therefore there must exist another key in self.__encounter_dict with key "2-1-4"
			# the values at "1-2-4" and "2-1-4" represent the sampled subprofiles of "1" and "2" respectively
			# we can thus calculate a similarity for that encounter
			# NOTE that self.__encounter_dict necessarily has an even number of keys
			if self.__hide_parallel:
				encounterkey_sim_pairs = self.create_encounterkey_sim_pairs_inparallel()
			else:
				encounterkey_sim_pairs = [self.create_encounterkey_sim_pairs(encounterkey) for encounterkey \
										in self.__initial_snapshot_refs.keys() if "-" in encounterkey]

			# Assign key-value pairs to self.__sim_dict
			for encounterkey, sim in encounterkey_sim_pairs:
				self.__sim_dict[encounterkey] = sim
		"""

		# PICKLE self.__sim_dict if a sim_pickle_string is given
		if self.__sim_dict_pickle_string is not None:
			self.pickle_sim_dict()
		# PICKLE self.__initial_snapshot_refs
		if self.__initial_snapshot_pickle_string is not None:
			self.pickle_initial_snapshot_refs()

		# submit the self.__sim_dict to the mobility_model of graph
		self.graph.mobility_model.set_sim_dict(self.__sim_dict)


	def initialize_dynamic_percentile_dict(self):
		if self.__peers is None:
			self.initialize_peers()

		# initialize self.dynamic_percentile_dict as key-value pairs of peer:self.__dynamic_percentile
		# yet only if it has not been initialized yet
		if self.dynamic_percentile_dict is None:
			self.dynamic_percentile_dict = {peer:self.__dynamic_percentile for peer in self.__peers}

	def initialize_time_horizon_dict(self):
		if self.__peers is None:
			self.initialize_peers()

		# initialize self.time_horizon_dict as key-value pairs of peer:self.__time_horizon
		# yet only if it has not been initialized yet
		if self.time_horizon_dict is None:
			self.time_horizon_dict = {peer:self.__time_horizon for peer in self.__peers}


	def pickle_sim_dict(self):
		""" Create a pickle file of the calculated similarities in the sim_dict. The parameters that influence the similarities are

		"""
		pickle_path = os.path.join(self.sim_dict_pickle_dir, self.__sim_dict_pickle_string)
		# create the output path if it does not exist
		if not os.path.isdir(self.sim_dict_pickle_dir):
			os.mkdir(self.sim_dict_pickle_dir)

		with open(pickle_path, mode = "wb") as f:
			pickle.dump(self.__sim_dict, f)

		print("Pickled sim_dict to {}".format(pickle_path))


	def pickle_initial_snapshot_refs(self):
		""" Create a pickle file of the references to the initial snapshots in the self.__train_df.
		These row indices are meant to serve as lookup (keys) for faster reference.
		"""
		pickle_path = os.path.join(self.initial_snapshot_refs_pickle_dir, self.__initial_snapshot_pickle_string)

		with open(pickle_path, mode = "wb") as f:
			pickle.dump(self.__initial_snapshot_refs, f)

		print("Pickled initial_snapshot_refs to {}".format(pickle_path))



	def initialize_min_sim_to_sender_dict(self):
		""" Initialize the min_sim_to_sender_dict such that holds a min_sim_to_sender threshold for every
			peer (userId) of self.__peers above which data from the child (sender) are received.

			dynamic: (self.__min_sim_to_sender_dynamic = True)
			static: (self.__min_sim_to_sender_dynamic = False)
				either for a specified threshold self.min_sim_to_sender = some float
				or     for an implicitly specified threshold via self.__dynamic_percentile

		"""
		# dynamic thresholding
		if self.__min_sim_to_sender_dynamic == True:
			if self.__sim_dict is None:
				self.initialize_sim_dict()

			# initialize min_sim_to_sender_dict
			self.min_sim_to_sender_dict = {peer : self.get_percentile_sim(peer) for peer in self.__peers}
		# static thresholding
		else:
			if self.__peers is None:
				self.initialize_peers()

			if self.__dynamic_percentile is None:
				# cast self.min_sim_to_sender to a dictionary holding for peer keys the corresponding min_sim_to_sender threshold (constant)
				print("'None' percentile specified. Using static predefined <min_sim_to_sender> threshold = {}.".format(self.min_sim_to_sender))
				self.min_sim_to_sender_dict = { peer : self.min_sim_to_sender for peer in self.__peers}
			else:
				print("Percentile {} specified. Using global static <min_sim_to_sender> threshold = {}.".format(self.__dynamic_percentile, self.get_percentile_sim_global()))
				self.min_sim_to_sender_dict = { peer : self.get_percentile_sim_global() for peer in self.__peers}



	#####################################################
	# 5. FILL_SNAPSHOTS FUNCTIONS [MAIN FUNCTIONALITY]  #
	#####################################################



	def fill_snapshots(self):
		""" Fill all snapshots in the future_snapshots_dict for all peers. Requires initial snapshots.
		Only the snapshots at time t=0 hold references to train_df, the rest holds references to future_snapshots.

		Consecutive snapshots at times t>0 only hold profiles that have been exchanged.

		NOTE If inparallel == True, fill snapshots in parallel using the multiprocessing modules Pool workers.

		NOTE It is possible that the peer receives his/her own original profile after consecutive data exchanges.
		If you want to this to not be the case:
			TODO: filter payloads in this function


		I. Initialize simulation (snapshots, pairwise similarities ...)
		II. Iterate over all epochs t = 1,..,T and all peers
			II.1 For_loop (epoch t fixed) [can be parallelized over]
				II.1.1 Collect the payload [i.e. information to be shared by the children with the parent] over all children of a parent node
				II.1.2 Aggregate the information in the payload [if specified]
				II.1.3 Run a garbadge collection of the aggregated information in order to prevent for example duplicates
			II.2 [OPTIONAL] Update system parameters; else use static parameters
			II.3 Merge the aggregated information with the currently available information in the snapshot (future_snapshots_dict[peer])
			II.4 Every save_every_i periods, save all snapshots to a designated output_dir, where every future snapshot is saved to a distinct .csv file
				NOTE that the train_df is only needed at training time
		III. Stopping criterion; stop the iteration in II. as soon as the criterion is fulfilled.

		"""
		#############################
		# I. Initialize experiment  #
		#############################
		self.initialize_experiment()


		if self.__processes is not None:
			# start a worker pool
			with Pool(processes=self.__processes) as pool:

				lower_while_bound = 1
				upper_while_bound = max(self.time_horizon_dict.values())

				# While Loop
				#################
				while lower_while_bound <= upper_while_bound:

					##########################################
					# II. Outer for loop;
					##########################################
					for t in range(lower_while_bound, upper_while_bound+1,1):
						# track the current time globally
						self.t = t
						print("Calculate t = {}/{}".format(self.t, upper_while_bound), end = "\r")
						# initialize the previously calculated future snapshots
						new_future_snapshots_dict_entries = []

						# generate graph at epoch t
						# create vertices for all <peer>s at epoch <self.t>
						for peer in self.__peers:
							self.graph.add_vertex(peer+":"+str(self.t))
						# yet, only add edges for those that seek connections
						peers_seeking_connections_in_t = [peer for peer, time_horizon in self.time_horizon_dict.items() if self.t <= time_horizon]
						peer_counter_df = pd.DataFrame([self.graph.mobility_model.N for _ in peers_seeking_connections_in_t], index = peers_seeking_connections_in_t, columns = ["counter"])
						#
						self.graph, _ = self.graph.mobility_model.generate_graph_at_t(self.graph, self.t, peer_counter_df, self.graph.mobility_model.random_gen)


						# II.1 Inner for-loop; for all peers
						#	|____II.1.1 Collect the payload
						#	|____II.1.2 Aggregate payload
						#	|____II.1.3 Collect garbadge
						######################################
						#WORKED: June 21, 2021
						results_object = pool.map_async(self.inner_for_loop, product([self.t], self.__peers, [self.timestamp_delta]))
						results_object.wait()

						new_future_snapshots_dict_entries = results_object.get()

						# [OPTIONAL] II.2 Update system parameters
						#######################################
						if self.parameter_control_model is not None:
							# update parameters: dynamic_percentile (theta) and self.graph (T)
							self.parameter_control_model.update_parameters(t, new_future_snapshots_dict_entries)

						# II.3 Update self.future_snapshots_dict
						#################################################
						# after all new future snapshots_dict entries have been calculated
						# persist to self.future_snapshots_dict
						for peer, new_future_snapshots_dict_entry in zip(self.__peers, new_future_snapshots_dict_entries):
							self.future_snapshots_dict[peer] = new_future_snapshots_dict_entry
						# ALTERNATIVE:
						#self.future_snapshots_dict[peer] = dict(zip(self.__peers, new_future_snapshots_dict_entries))


						# II.4 Save parameter logs and snapshots every <self.__save_every_i> epochs
						#######################################
						# II.4.A Save paramer_log
						self.save_parameters()

						# II.4.B Check Stopping Criterion
						if self.stopping_criterion():
							# Stopping procedure
							self.stopping_procedure()
							return
						else:
							# save the snapshot every <save_every_i>
							if self.t % self.__save_every_i == 0:
								self.save_future_snapshots()



					# update the lower and upper boundaries of the outer for-loop
					lower_while_bound = upper_while_bound+1
					upper_while_bound = max(self.time_horizon_dict.values())


		else:
			lower_while_bound = 1
			upper_while_bound = max(self.time_horizon_dict.values())

			# While Loop
			#################
			while lower_while_bound <= upper_while_bound:

				##########################################
				# II. Outer for loop;
				##########################################
				# NOTE that the lower and upper while bounds will be updated after each while loop!
				# This is done because the global time horizon (max(self.time_horizon_dict.values())) can increase(!),
				# see for instance DistributedGradientTracking
				for t in range(lower_while_bound, upper_while_bound+1, 1):
					# track the current time globally
					self.t = t
					print("Calculate t = {}/{}".format(self.t, upper_while_bound), end = "\r")
					new_future_snapshots_dict_entries = []

					# generate graph at epoch t
					# create vertices for all <peer>s at epoch <self.t>
					for peer in self.__peers:
						self.graph.add_vertex(peer+":"+str(self.t))
					# yet, only add edges for those that seek connections
					peers_seeking_connections_in_t = [peer for peer, time_horizon in self.time_horizon_dict.items() if self.t <= time_horizon]
					peer_counter_df = pd.DataFrame([self.graph.mobility_model.N for _ in peers_seeking_connections_in_t], index = peers_seeking_connections_in_t, columns = ["counter"])
					#
					self.graph, _ = self.graph.mobility_model.generate_graph_at_t(self.graph, self.t, peer_counter_df, self.graph.mobility_model.random_gen)

					# II.1 Inner for-loop; for all peers
					#	|____II.1.1 Collect the payload
					#	|____II.1.2 Aggregate payload
					#	|____II.1.3 Collect garbadge
					# for all peers, create the new future snapshots dict entry that presents the knowledge of a peer
					# ater t periods in the network
					for peer in self.__peers:
						t_and_peer = [self.t, peer, self.timestamp_delta]
						new_future_snapshots_dict_entry = self.inner_for_loop(t_and_peer)
						new_future_snapshots_dict_entries.append(new_future_snapshots_dict_entry)

					# [OPTIONAL] II.2 Update system parameters
					#######################################
					if self.parameter_control_model is not None:
						# update parameters: dynamic_percentile (theta) and self.graph (T)
						self.parameter_control_model.update_parameters(self.t, new_future_snapshots_dict_entries)

					# II.3 Update self.future_snapshots_dict
					#################################################
					# after all new future snapshots_dict entries have been calculated
					# persist to self.future_snapshots_dict
					for peer, new_future_snapshots_dict_entry in zip(self.__peers, new_future_snapshots_dict_entries):
						self.future_snapshots_dict[peer] = new_future_snapshots_dict_entry


					# II.4 Save parameter logs and snapshots every <self.__save_every_i> epochs
					#######################################
					# II.4.A Save paramer_log
					self.save_parameters()

					# II.4.B Check Stopping Criterion
					if self.stopping_criterion():
						# Stopping procedure
						self.stopping_procedure()
						return
					else:
						# save the snapshot every <save_every_i>
						if self.t % self.__save_every_i == 0:
							self.save_future_snapshots()

				# update the lower and upper boundaries of the outer for-loop
				lower_while_bound = upper_while_bound+1
				upper_while_bound = max(self.time_horizon_dict.values())

		return


	def inner_for_loop(self, t_and_peer):
		""" This is the inner for loop (for parallelization) over the peers and time periods in the fill_snapshots method.
		Input:
			- t_and_peer : <list> of
							[0]: <str> id of the userId
							[1]: <int> of the time step
							[2]: <int> of the timestamp_delta (default: None)
			-
		Output:
			- new_future_snapshots_dict_entry :  [to be referenced in self.future_snapshots_dict] outside the worker Pool's work

		NOTE The new_future_snapshots_dict_entry has to be persisted in self.future_snapshots_dict outside the loops for reasons of thread-safety.
		"""
		t    = t_and_peer[0]
		peer = t_and_peer[1]
		timestamp_delta = t_and_peer[2]

		# parent = data receiver; child = data sender; child_child = data sender that sent data to child in the past (in particular child is a child_child)
		parent        = peer
		parent_at_t   = peer + ":" + str(t)
		children      = [el.split(":")[0] for el in self.graph.children(parent_at_t)]

		# COLLECT PAYLOADS (list of pandas.DataFrame) from all children except the parent
		#####################################################################################
		# payloads : list of pandas dataframes to be aggregated; every dataframe represents data to use for aggregation by a distinct child
		payloads = [self.collect_payload(parent, child, t, timestamp_delta) for child in children if child != parent]# perhaps the if statement can be removed due to changes in graph.py

		# AGGREGATE PAYLOADS
		#######################
		# list of list of pandas.DataFrame (rows) that hold aggregated rating information
		# per sender
		aggregated_payloads = [self.aggregate_payload(payload, peer) for payload in payloads]

		# add the existing snapshots (time : <t) to the aggregated_payload_list (time : t)
		old_and_new_parts = [self.future_snapshots_dict[parent]] + aggregated_payloads
		# concatenate old_and_new_parts
		new_future_snapshots_dict_entry = self.fast_concat(old_and_new_parts)

		# COLLECT GARBADGE
		########################
		# collect garbadge: in this case duplicates and profiles that are not within the top keepN
		new_future_snapshots_dict_entry = self.collect_garbadge(new_future_snapshots_dict_entry)

		return new_future_snapshots_dict_entry


	def stopping_criterion(self, dynamic_percentile_threshold = 1.0):
		"""
			Check whether to stop the simulation (early).

			The simulation should stop if
				(a) the local time horizons are all lower than the current epoch (no connections are established)
				(b) the dynamic_percentiles are all max. (=1.0) (no payloads are exchanged in connections)
		"""
		a = max(self.time_horizon_dict.values()) <= self.t #TODO: TEST "<="; original: "<"                                      # ? all time horizons smaller than the current epoch?
		b = min(self.dynamic_percentile_dict.values()) >= dynamic_percentile_threshold # ? all dynamic percentiles larger or equal <dynamic_percentile_threshold>
		if a | b :
			return True # stop
		else:
			return False # continue

	def stopping_procedure(self):
		"""
		 	Run this procedure as soon as the simulation has been stopped (see (III) in <self.inner_for_loop>)
		"""
		self.save_future_snapshots(early_stop= True)

		return


	#####################################################
	# 6. ABSTRACT METHODS (REQUIRE OVERRIDING)          #
	#####################################################

	# 6. II.1.1 COLLECT PAYLOAD
	@abstractmethod
	def collect_payload(self, parent, child, t, timestamp_delta):
		""" Collect the information that child is willing to share with parent.

		Input:
			- child : <str> ID of the sending end of the data exchange
			- parent: <str> ID of the receiving end of the data exchange

		Output:
			- payload: pandas.DataFrame that holds the information going to be sent from the child to the parent

		NOTE that the meaning of "userId" unlike to the aggregating version refers to the 'identifiable' originator of the data,
		whereas the "sim" is the similarity of the originator to the data tenant [given by the dictionary key of self.future_snapshots_dict.
		"""
		return self.default_snapshot_row()


	# 6. II.1.2 AGGREGATE PAYLOAD
	@abstractmethod
	def aggregate_payload(self, payload):
		""" Aggregate a payload (pandas.DataFrame as the output of collect_payload) into a list of pandas.DataFrames for future concatenation
			into the new_future_snapshots_dict_entry.

		Input :
			- payload (pandas dataframe; output of <collect_payload>
				NOTE a payload is the raw information that should be aggregated by a single sender (child)
		Output:
			- aggregated_payload_list : list of pandas.DataFrame
		"""
		return [payload]



	# 6. II.1.3 COLLECT GARBADGE
	@abstractmethod
	def collect_garbadge(self, snapshots_dict_entry):
		""" Remove unnecessary information from the new_future_snapshots_dict_entry. """
		return snapshots_dict_entry






if __name__ == "__main__":

	#datapath  = os.path.join(root_dir,"data/ml-latest-small/ratings.csv")
	datapath = os.path.join(root_dir,"data/test")
	n_splits = 5
	random_state = 426 # for train/test
	N            = 3
	#topN         = 3
	T            = 2
	random_seed  = 51423 # for graph generation
	#make_pickle  = True
	#sim_string   = "cosine"


	# LOAD MOVIELENS data (100K)
	################################
	dataset = os.path.split(datapath)[1]
	print("Read in data : {}...".format(datapath))
	# read the data in the form of a table with columns
	# user id 	item id 	rating 	   timestamp; where timestamp will not be used
	#                       np.float64
	rating_df = pd.read_csv(datapath)
	# drop timestamp
	rating_df = rating_df.drop(["timestamp"], axis = 1)
	# cast the "userId" to type str
	rating_df["userId"] = rating_df["userId"].astype(str)
	# cast "movieId" to "itemId"
	rating_df = rating_df.rename(columns= {"movieId":"itemId"})
	# SPLIT TRAIN/TEST SET
	#################################
	kf = KFold(n_splits=n_splits, random_state = random_state, shuffle = True)
	folds = []
	for train_index, test_index in kf.split(rating_df):
		train_df, test_df = rating_df.loc[train_index,:], rating_df.loc[test_index,:]
		folds.append((train_df, test_df))
	# peer_list (<list> of <str>); the order is reproducible, and only depends on the input dataset (and train/test split)
	peers = rating_df["userId"].drop_duplicates().tolist()
	print("Build random graph ...")
	# initialize a graph object that will be a template for all k folds
	graph = DAG(graph_dict = None, peers = peers, T = None)
	# initialize the mobility model
	mobility_model = AssignNMobility(graph, N, T, random_seed = random_seed)
	# BUILD RANDOM GRAPH - ONLY HAS TO BE DONE ONCE; THEN COPIED
	################################
	mobility_model.generate_graph()
	graph = mobility_model.graph()

	fold = folds[0]
	# unpack train and test set [of the fold]
	train_df, test_df = fold
	initial_snapshot_refs = df2Snapshot_refs(train_df, peers)


	template = DecAlgoTemplate(graph, train_df, initial_snapshot_refs, snapshot_info_string = "", dataset_string = "", output_dir = os.path.join(root_dir,"data/snapshots"),\
						save_every_i = 5, sim_string = "Pearson")

	#template.map_userIds()
	#template.map_itemIds()

	template.initialize_sim_dict()
	#print(template.userId_dict())
	#print(template.itemId_dict())
	#print(template.sim_dict())
