import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)

# for debugging

# for production
from algorithms.templates.DecAlgoTemplate import DecAlgoTemplate
from algorithms.templates.CollectPayloadTemplate import CollectPayloadTemplate
from algorithms.templates.AggregatePayloadTemplate import AggregatePayloadTemplate
from algorithms.templates.CollectGarbadgeTemplate import CollectGarbadgeTemplate


from mobility_models.graph import DAG
from mobility_models.TIJMobility import TIJMobility
from mobility_models.DirectNMobility import DirectNMobility
from mobility_models.AssignNMobility import AssignNMobility
from mobility_models.BarabasiAlbertNMobility import BarabasiAlbertNMobility
from mobility_models.WattsStrogatzNMobility import WattsStrogatzNMobility
from mobility_models.NeighborhoodFormationNMobility import NeighborhoodFormationNMobility
from mobility_models.UniformRandomNMobility import UniformRandomNMobility
from parameter_control.StaticParameters import StaticParameters
from parameter_control.DistributedGradientTracking import DistributedGradientTracking
from parameter_control.QuickstopParameters import QuickstopParameters

from utilities.util import load_ratings_df, make_snapshot_info_string, build_execution_graph, make_train_test_split, make_train_test_split_per_user, \
						make_sim_dict_pickle_string, make_initial_snapshot_string, make_uniform_percentile_dict, make_uniform_T_dict
import numpy as np
import argparse
import pickle


class DecShCFv3(CollectPayloadTemplate, AggregatePayloadTemplate, CollectGarbadgeTemplate, DecAlgoTemplate):
	""" This class is meant to implement the decentralized Collaborative Filtering (CF) algorithm in which transaction pseudonyms are used and
	a portion of profile entries are hidden in each and every encounter/for a given graph (from the graph.DAG class).

	In version3, we implement a version of the aggregation mechanism in Shokri et al. (2008). The aggregation happens as follows:
	The payload which constitutes topN (partially hidden) profiles is sorted by the similarity of the child to the child_children.
	This means that during the contact, child does not need to calculate similarities between all profiles in his database to the parent.
	In order to unbias the perfect similarity (sim(child,child) = 1) between child (as child) and child as (child_child), we substitute it
	with the similarity between child and parent, which has to be calculated anyways.

	DecShCFv2
	    |_________sim between ___________     for sorting the topN profiles that form the payload: consequently, the difference between v2 and v3 is
		|                                |                                                       : limited to the <collect_payload> method
	    |        DecShCFv3	             |
		|           |                    |
		|  _________|_                   |
		| |           |sim between       |
	    | |           |                  |
	    | |           |                  |
	child_child1-     |             |    |
	              \   |             |    |
	child_child2--- child (sender) -|- parent (receiver)
	              /                 |
	child_child3-                   |
	################################|##############################
	the profiles on this side       | the parent profile is shared between child
	are held by child before sim.   | in a contact and is available to the child
	comparison                      | after the sim. comparison




	In particular, in the original version, it was possible that the sender sends multiple (unaggregated) ratings for the same item

	topN (most similar to parent(receiver))
	- Take the ratings by child_child1
	- Take the ratings by child_child2 that are not already included in the above
	- And so on ...

	The resulting data is thus in essence a neighborhood profile (representing the topN child_children (via child, which might be among the child_children))
	in contrast to DecShCF where the data is separable. Thus the code of DecShCFv2 is only different in the positions marked with (x)


	The graph has to be built with a mobility model. Then the method <fill_snapshots>/<fill_snapshots_inparallel>
	will successively calculate the snapshots until a final time period T. At given timeintervals the snapshots will be able to be saved.

	Inheriting from DecAlgoTemplate, not much has to be coded.

	II.1.1 collect_payload (including fill_in_missing_columns)
	II.1.2 aggregate_payload
	II.1.3 collect_garbadge



	"""


	def __init__(self, graph, train_df, snapshot_info_string = "", dataset_string = "", initial_snapshot_string = "", output_dir =  os.path.join(root_dir,"data/snapshots"),\
							save_every_i = 5, sim_string = "cosine", min_sim_to_sender = 0.0, min_sim_to_child_child = 0.0, topN = 3, \
							max_agg_sim = 100, sim_mat_path = None, min_sim_to_sender_dynamic = False, dynamic_percentile = None, time_horizon = None, \
							hide_p = None, hide_seed = None, sim_dict_pickle_string = None, timestamp_delta = None, processes = None,\
							mean_centering = False, dynamic_percentile_dict = None, time_horizon_dict = None, anonymous_contacts = False):

		DecAlgoTemplate.__init__(self, graph, train_df, snapshot_info_string = snapshot_info_string, dataset_string = dataset_string, initial_snapshot_string = initial_snapshot_string, output_dir = output_dir,\
							save_every_i = save_every_i, sim_string = sim_string, sim_mat_path = sim_mat_path, min_sim_to_sender = min_sim_to_sender, \
							min_sim_to_child_child = min_sim_to_child_child, topN = topN, min_sim_to_sender_dynamic = min_sim_to_sender_dynamic, \
							dynamic_percentile = dynamic_percentile, time_horizon = time_horizon, hide_p = hide_p, hide_seed = hide_seed, sim_dict_pickle_string = sim_dict_pickle_string,\
							timestamp_delta = timestamp_delta,  processes = processes, mean_centering = mean_centering,	dynamic_percentile_dict = dynamic_percentile_dict,\
							time_horizon_dict = time_horizon_dict, anonymous_contacts= anonymous_contacts)

		# for persisting snapshot information to disk
		self.algo_string 				= "DecShCFv3"

	# 6. II.1.1 COLLECT PAYLOAD
	# inherited from CollectPayloadTemplate

	# 6. II.1.2 AGGREGATE PAYLOAD
	# inherited from AggregatePayloadTemplate

	# 6. II.1.3 COLLECT GARBADGE
	# inherited from CollectGarbadgeTemplate



if __name__ == "__main__":
	##############
	# PARSE ARGS #
	##############
	parser = argparse.ArgumentParser()
	parser.add_argument("datapath"		, help="path to the user-item matrix; e.g. ratings.csv in ml-latest-small" 		, action = "store")
	parser.add_argument("experiment_name", help="name of the simulation experiment, the resulting snapshots will be persisted to \
											./data/snapshots/<experiment_name>", action = "store")
	parser.add_argument("N"				, help="number of contacts per user and epoch.", action = "store")
	parser.add_argument("T"				, help="number of epochs to run simulation" 									, action = "store")
	parser.add_argument("graph_seed"	, help="random seed for graph generation"				 						, action = "store")
	parser.add_argument("traintest_seed", help="random seed for test-train splitting" 									, action = "store")
	parser.add_argument("payload_size"	, help="number of profiles per payload (topN)" 									, action = "store")
	parser.add_argument("mobility_model", help="name of the mobility model to use for graph generation"					, action = "store")
	parser.add_argument("percentile"	, help="percentile of similarity distribution only above which payloads are sent; e.g.\
												if percentile = 0.9; a user only sends a payload, if the similarity is in the top 10% of all similarities", action = "store")
	parser.add_argument("--ttsplit_per_user"	, help="if '0': sample training a test sets over all ratings by all users (80-20),\
												if '1': split the training and testing sets on a per user basis. default: '1'", action = "store")
	parser.add_argument("--static_percentile", help="if '1': the percentile is interpreted as a global percentile, in other words, the similarity threshold\
												above which a user sends a payload is equal for all users. \
												default: every user has an individual threshold based on his/her individual similarity distribution and <percentile> to all other peers \
												above which he/she sends a payload." 		, action = "store")
	parser.add_argument("--min_sim_to_sender", help="minimum threshold only above which payloads are exchanged in a contact; default: 0.0" 		, action = "store")
	parser.add_argument("--min_sim_to_child_child", help="minimum threshold only above which profiles are considered to be added to the payload; default: 0.0" 		, action = "store")
	parser.add_argument("--sim_mat_path", 	help="path to a similarity_matrix to use." 		, action = "store")
	parser.add_argument("--timestamp_delta"			, help="delta epochs beyond which not to consider entries for aggregation", action = "store")
	parser.add_argument("--sim_string",	help="name of the similarity metric to use; one of ['cosine',...]; default: 'cosine'", action = "store")
	parser.add_argument("--dataset_string", help="custom name of the dataset; default: filename of datapath", action = "store")
	parser.add_argument("--anonymous_contacts", help="if '1': contacts are anonymous and it is not possible to distinguish users between epochs")
	parser.add_argument("--max_agg_sim", help="maximum aggregated similarity above which aggregated ratings are not further aggregated", action = "store")
	parser.add_argument("--sim_dict", help="precalculated pickled similarity dict located in the ''./data/similarity_dicts' directory (!), format {'peer1:peer2':sim(peer1,peer2),...}", action = "store")
	parser.add_argument("--m", help="parameter used for sampling random graphs. e.g. for BarabasiAlbertNMobility we add nodes with m edges each randomly.", action = "store")
	parser.add_argument("--k", help="parameter userd for sampling random graphs. e.g. for WattsStrogatzNMobility we start with a ring of nodes connected to k other nodes in a ring.", action = "store")
	parser.add_argument("--p", help="parameter userd for sampling random graphs. e.g. for WarrsStrogatzNMobility, we rewire edges with probability p.", action = "store")
	parser.add_argument("--processes", help="<int> parameter (>1) that specifies the number of processes for parallel computation; contacts within an epoch are calculated in parallel; DEFAULT: None", action = "store")
	parser.add_argument("--mean_centering", help="'1': center all rating profiles such that they have mean rating = 0; DEFAULT: no mean centering", action = "store")
	parser.add_argument("--param_control", help="string that defines how system parameters are controlled. default: 'static'")
	parser.add_argument("--random_percentile", help="test", action = "store")
	parser.add_argument("--random_T", help="test", action = "store")
	parser.add_argument("--tij_datapath", help="specify the path to the tij dataset to use in combination with the TIJMobility mobility model", action = "store")
	parser.add_argument("--tij_timestamp_aggregation", help="specify the number of timestamps to aggregate into an epoch", action = "store")

	args = parser.parse_args()

	# default paths
	sim_dict_pickle_dir = os.path.join(root_dir,"data/similarity_dicts")
	snapshot_dir		= os.path.join(root_dir,"data/snapshots")

	# deprecated parameters
	hide_seed = 104
	hide_p =	None

	# default values
	n_splits = 5
	foldNr = 0

	# positional arguments
	datapath				= args.datapath
	experiment_name			= args.experiment_name
	N						= int(args.N)
	T						= int(args.T)
	graph_seed				= int(args.graph_seed)
	traintest_seed			= int(args.traintest_seed)
	topN					= int(args.payload_size)
	mobility_string			= args.mobility_model
	percentile 				= args.percentile
	if percentile == "None":
		percentile = None
	else:
		percentile = float(args.percentile)

	# optional arguments
	if args.static_percentile == '1':
		min_sim_to_sender_dynamic =  False
	else:
		min_sim_to_sender_dynamic =  True	# default

	if args.min_sim_to_sender is not None:
		min_sim_to_sender = float(args.min_sim_to_sender)
	else:
		min_sim_to_sender = 0.0				# default

	if args.min_sim_to_child_child is not None:
		min_sim_to_child_child = float(args.min_sim_to_child_child)
	else:
		min_sim_to_child_child = 0.0		# default

	if args.sim_mat_path is not None:
		sim_mat_path = args.sim_mat_path
	else:
		sim_mat_path = None					# default

	if args.timestamp_delta is not None:
		timestamp_delta = int(args.timestamp_delta)
	else:
		timestamp_delta = None 				# default

	if args.sim_string is not None:
		sim_string = args.sim_string
	else:
		sim_string = "cosine"				# default

	if args.dataset_string is not None:
		dataset_string = args.dataset_string
	else:
		dataset_string = os.path.split(datapath)[1]	# default

	if args.anonymous_contacts == '1':
		anonymous_contacts = True
	else:
		anonymous_contacts = False			# default

	if args.max_agg_sim is not None:
		max_agg_sim = float(args.max_agg_sim)
	else:
		max_agg_sim = 100					# default

	if args.sim_dict is not None:
		sim_dict_pickle_string = args.sim_dict
		sim_string = sim_dict_pickle_string

		sim_dict_pickle_path = os.path.join(sim_dict_pickle_dir, sim_dict_pickle_string)

		with open(sim_dict_pickle_path, mode = "rb") as f:
			sim_dict = pickle.load(f)
	else:
		sim_dict_pickle_string = make_sim_dict_pickle_string(dataset_string, traintest_seed, graph_seed, hide_seed, hide_p, T, sim_string)
		sim_dict = None


	if args.ttsplit_per_user == '0':
		ttsplit_per_user = False
	else:
		ttsplit_per_user = True #default

	if args.m is not None:
		m = int(args.m)
	else:
		m = None

	if args.k is not None:
		k = int(args.k)
	else:
		k = None

	if args.p is not None:
		p = float(args.p)
	else:
		p = None

	if args.processes is not None:
		processes = int(args.processes)
	else:
		processes = None

	if args.mean_centering == '1':
		mean_centering = True
	else:
		mean_centering = False # default

	if args.param_control is not None:
		parameter_control_string = args.param_control
	else:
		parameter_control_string = "static" # default


	# make randomized dynamic_percentile_dict if specified
	if args.random_percentile == '1':#is not None:
		random_percentile_string = "uniform"
		# if a dynamic_percentile_dict is specified, we do not have to specify percentile (=None)
		percentile = None
	else:
		percentile_dict = None
		random_percentile_string = None

	# make randomized time horizon T, if specified
	if args.random_T == '1':#is not None:
		random_T_string = "uniform"
		# if a T_dict is specified, we do not have to specify percentile (=None)
		T = None
	else:
		T_dict = None
		random_T_string = None

	# if TIJMobility is specified, a path to the tij dataset to use it required.
	if args.tij_datapath is not None:
		tij_datapath = args.tij_datapath
	else:
		tij_datapath = None

	if args.tij_timestamp_aggregation is not None:
		tij_timestamp_aggregation = int(args.tij_timestamp_aggregation)
	else:
		tij_timestamp_aggregation = None

	#
	output_dir 				= os.path.join(snapshot_dir,experiment_name)



	######################################
	# DECENTRALIZED CF with Aggregation  #
	######################################


	# load rating data as a pandas.DataFrame in coordinate format with the columns ["userId", "itemId", "rating"]
	rating_df = load_ratings_df(datapath, userId_label = "userId", itemId_label = "movieId", rating_label = "rating", timestamp_label = "timestamp")#ml_smallest_datapath)

	# make train/test split
	if ttsplit_per_user:
		train_frac = (n_splits-1)/n_splits
		train_df, test_df = make_train_test_split_per_user(rating_df, train_frac = train_frac, traintest_seed = traintest_seed)
	else:
		train_df, test_df = make_train_test_split(rating_df, n_splits = n_splits, traintest_seed = traintest_seed, shuffle = True, foldNr = foldNr)#

	# extract unique peers
	peers = rating_df.loc[:,"userId"].drop_duplicates().values

	# randomize percentiles in percentile_dict if desired
	if random_percentile_string is not None:
		if random_percentile_string == "uniform":
			lower_percentile_bound = 0.7 #percentile not theta!!! 1-theta
			upper_percentile_bound = 0.9
			percentile_dict = make_uniform_percentile_dict(peers, lower_percentile_bound, upper_percentile_bound, random_seed = traintest_seed)

	# randomize T in T_dict if desired
	if random_T_string is not None:
		if random_T_string == "uniform":
			lower_T_bound = 25 #
			upper_T_bound = 75
			T_dict = make_uniform_T_dict(peers, lower_T_bound, upper_T_bound, random_seed = graph_seed)


	# initialize a graph and a mobility model with peers from rating_df (userId's)
	graph			= DAG(peers=peers)

	if mobility_string == "AssignNMobility":
		mobility_model = AssignNMobility(graph, N, T, graph_seed=graph_seed)
		save_every_i = 5
	elif mobility_string == "DirectNMobility":
		mobility_model = DirectNMobility(train_df, N, T, sim_string = sim_string, min_sim = min_sim_to_sender, sim_dict = sim_dict, graph_seed = graph_seed)
		save_every_i = 5
		mobility_string = "DirectNMobility"
	elif mobility_string == "NeighborhoodFormationNMobility":
		mobility_model = NeighborhoodFormationNMobility(train_df, N, sim_string = sim_string, sim_dict = sim_dict)
		save_every_i = 1
		mobility_string = "NFNMobility" # shorter name
	elif mobility_string == "UniformRandomNMobility":
		T = 1
		mobility_model = UniformRandomNMobility(graph, N, T, graph_seed=graph_seed)
		save_every_i = 1
	elif mobility_string == "BarabasiAlbertNMobility":
		if m is None:
			print("PLEASE DO NOT FORGET TO SPECIFY THE PARAMETER M VIA THE '--m' FLAG FOR THE BARABASI-ALBERT MOBILITY MODEL.")
		mobility_model = BarabasiAlbertNMobility(graph, N, T, m, graph_seed = graph_seed)
		save_every_i = 5
		mobility_string += "m={}".format(m)
	elif mobility_string == "WattsStrogatzNMobility":
		if (k is None) | (p is None):
			print("PLEASE DO NOT FORGET TO SPECIFY THE PARAMETERS k AND p VIA THE '--k' AND '--p' FLAGS FOR THE WATTS-STROGATZ MOBILITY MODEL.")
		mobility_model = WattsStrogatzNMobility(graph, N, T, k, p, graph_seed = graph_seed)
		save_every_i = 5
		mobility_string += "k={}p={}".format(k,p)
	elif mobility_string == "TIJMobility":
		mobility_model = TIJMobility(graph, tij_datapath, tij_timestamp_aggregation)
		save_every_i = 5
		#mobility_string += "tij={}".format(os.path.split(tij_datapath)[-1])


	# link mobility_model to graph
	graph.use_mobility_model(mobility_model)

	#####################################
	# CREATE SNAPSHOT_INFO_STRING       #
	#####################################
	snapshot_info_string = make_snapshot_info_string(topN, sim_string, dataset_string, graph_seed, n_splits, traintest_seed, N, \
												mobility_string, percentile, hide_p = hide_p, timestamp_delta = timestamp_delta, min_sim_to_sender_dynamic = min_sim_to_sender_dynamic,\
												ttsplit_per_user = ttsplit_per_user, mean_centering = mean_centering, parameter_control_string = parameter_control_string, \
												anonymous_contacts = anonymous_contacts)

	#####################################
	# CREATE INITIAL SNAPSHOT STRING    #
	#####################################
	initial_snapshot_string = make_initial_snapshot_string(dataset_string, traintest_seed, ttsplit_per_user = ttsplit_per_user)

	# initialize a decentralized CF algorithm object that is able to run
	algorithm = DecShCFv3(graph, train_df, snapshot_info_string = snapshot_info_string, dataset_string = dataset_string, initial_snapshot_string = initial_snapshot_string, output_dir = output_dir, sim_string = sim_string,
								topN = topN, min_sim_to_child_child	= min_sim_to_child_child, min_sim_to_sender	= min_sim_to_sender, sim_mat_path = sim_mat_path,\
								min_sim_to_sender_dynamic = min_sim_to_sender_dynamic, dynamic_percentile = percentile, time_horizon = T, hide_seed = hide_seed, hide_p = hide_p,\
								sim_dict_pickle_string = sim_dict_pickle_string, timestamp_delta = timestamp_delta, save_every_i = save_every_i, processes = processes, \
								mean_centering = mean_centering, dynamic_percentile_dict = percentile_dict, time_horizon_dict = T_dict, anonymous_contacts = anonymous_contacts)

	########################################
	# SET PARAMETER CONTROL MODEL          #
	########################################
	if parameter_control_string == "static":
		parameter_control_model = StaticParameters(algorithm)
	elif parameter_control_string == "DGT":
		val_df = test_df
		alpha_dynamic_percentile = -0.1 # CAVEAT this "-" is important, since the percentiles are 1-theta!
		# note that the gamma in the paper is included in this, please refer to the DistributedGradientTracking class for further information
		beta_dynamic_percentile  = 0.1 # CAVEAT, here the "-" is not required since the consensus-term is symmetric and cancels out

		alpha_T     = 1.0 # default: 1.0
		beta_T      = 1.0 # default: 1.0
		epsilon     = 0.01

		parameter_control_model = DistributedGradientTracking(algorithm, val_df, alpha_dynamic_percentile, beta_dynamic_percentile, alpha_T, beta_T, epsilon)
	elif parameter_control_string == "QS":
		val_df = test_df
		epsilon     = 0.01
		parameter_control_model = QuickstopParameters(algorithm, val_df, epsilon)

	algorithm.use_parameter_control_model(parameter_control_model)

	# fill snapshots
	algorithm.fill_snapshots()
	#print(algorithm.future_snapshots_dict)
