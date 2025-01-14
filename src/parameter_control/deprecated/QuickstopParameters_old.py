
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)

from parameter_control.ParameterControlTemplate import ParameterControlTemplate

from utilities.util import convert_df_to_cornac_data

import pandas as pd

import cornac
from cornac.experiment import Experiment
from cornac.eval_methods import BaseMethod
from cornac.models import MF, SVD#, PMF, BPR, UserKNN, ItemKNN, MCF, WMF, SVD, MCF, WMF, VAECF, CVAECF, CVAE, GMF, MCF, NeuMF, ConvMF, CDL, HPF, FM, GlobalAvg, MMMF, MostPop
from cornac.metrics import RMSE#, MAE, Precision, Recall, NDCG, AUC, MAP
#from cornac.hyperopt import Discrete, GridSearch

class QuickstopParameters(ParameterControlTemplate):
	""" Class that defines a parameter control scheme in which users set their time horizon T to the current epoch if the performance differential is in (0,epsilon).
	 	Users thus stop entering into any further contact as soon as their performance 'saturates'.

		NOTE that theta is static (does not change over time).
		NOTE that T only changes once per user, once performance saturates.

		HERE performance is a performance metric of your choice before and after collection. Observe how non-collection yields only an averaging effect!
		HERE epsilon is a fixed positive float.
		HERE p is one of two parameters {theta: dissemination parameter, T: time horizon}
		HERE alpha_p and beta_p are parameter-specific weights. Probably alpha_theta = 0.1 and alpha_T = 1 are reasonable. As for beta_theta = beta_T = 0.5 seem to be good first trial values.

		Parameters:
			val_df :

		CAVEAT dynamic_percentile and theta are not (!) the same! Dynamic_percentile and theta are related by the following relationship:
			1-dynamic_percentile = theta
			Therefore, the signum of the innovation term in the (I) UPDATE FORMULA must be inverted for dynamic_percentile!

		"""
	#################################
	# 1. INITIALIZATION FUNCTIONS   #
	#################################
	def __init__(self, algorithm, val_df, epsilon, mobility_model = None):
		# call the template's initialization
		super().__init__(algorithm, val_df = val_df)

		self.epsilon = epsilon

		# parameter control string for future reference
		self.parameter_control_string	= "QS"

		# validation
		self.val_df = val_df
		self.val_dfs = None # initialize variable for <list> of dataframes per peer (peers might not be initialized when this parameter	control model is initialized)

		# mobility_model used for rewiring of pruned edges
		self.mobility_model = mobility_model

	def initialize_val_dfs(self):
		# split val_df into a list of sub_dfs ordered by self.algorithm.peers()
		self.val_dfs = [self.val_df.loc[self.val_df["userId"] == peer] for peer in self.algorithm.peers()]


	########################################################
	# 2. UPDATE SIMULATION PARAMETERS [MAIN FUNCTIONALITY] #
	########################################################

	def update_formula(self, time_horizon, epsilon, delta_performance, current_epoch):
		"""
		Set T = current_epoch only if performance saturates (delta_performance in (0,epsilon)).


		Example:

		"""
		# check saturation
		if delta_performance is None: # if no evaluation is possible, this indicates that the user still needs to collect data!
			new_time_horizon = time_horizon
		elif abs(delta_performance) == 0:
			new_time_horizon = time_horizon
		elif (0<abs(delta_performance)<epsilon): # if performance differentials are small (< epsilon); change parameters to lower collection
			new_time_horizon = current_epoch
		else: # if performance differentials are large; do not change parameters
			new_time_horizon = time_horizon
		return new_time_horizon


	def calculate_performance(self, k, train_data, test_data, use_bias, cornac_seed, model_string = "SVD", metric_string = "RMSE"):
		""" Calculate the performance of a recommender trained on train_df and evaluated on test_df. """
		if (len(train_data) == 0) | (len(test_data) == 0):
			return None

		# initialize evaluation method BaseMethod (from fixed train and test data)
		eval_method = BaseMethod.from_splits(train_data=train_data, test_data=test_data)
		# initialize matrix factorization model
		if model_string == "MF":
			model	= MF(     k=k, max_iter=50, learning_rate=0.01, lambda_reg=0.02, use_bias=use_bias, seed=cornac_seed)
		elif model_string == "SVD":
			model	= SVD(    k=k, max_iter=20, learning_rate=0.01, lambda_reg=0.02, early_stop=False, num_threads=0, trainable=True, init_params=None, seed=cornac_seed)

		models = [model]#, mcf_model]#, wmf_model, vaecf_model, cvaecf_model, cvae_model, gmf_model, neumf_model\
		         #, convmf_model, cdl_model, hpf_model, globalavg_model, mmmf_model, mostpop_model]#[mf_model, pmf_model, bpr_model, uknn_model, iknn_model]

		# initialize metrics to evaluate the models
		if metric_string == "RMSE":
			metric = RMSE()
		metrics = [metric]#MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

		# initialize experiment
		experiment = cornac.Experiment(eval_method=eval_method, models=models, metrics=metrics, user_based=True, verbose = False, save_dir = "./cornac_logs")
		# run experiment
		experiment.run()

		# extract performance [for an experiment, take the result of the 0-th model (in models), select the per-user results, for a specific metric]
		performance_dict = experiment.result[0].metric_user_results[metric_string]


		# extract the snapshot user from test_data; example: [('2', 7, 2.0)] (userId, itemId, rating) triples
		unique_test_users = set([el[0] for el in test_data])
		# if there is only one userId represented in the test_user_df, pick this userId
		if len(unique_test_users) == 1:
			snapshot_userId = test_data[0][0]
		# look-up which internalid (cornac-specific) the chosen userId has
		cornac_userId = experiment.eval_method.global_uid_map[snapshot_userId]

		# then select the performance of the user with the userId 0
		performance = performance_dict[cornac_userId] # the reference user has the userId 0

		return performance


	def calculate_delta_performance(self, k, train_data_old, train_data_new, test_data, use_bias, cornac_seed):
		""" Calculate a single delta performance, that is the difference in performance between recommenders on a training data entry in the
			<self.future_snapshots_dict> and the <new_future_snapshots_dict_entries>.
		"""
		old_performance = self.calculate_performance(k, train_data_old, test_data, use_bias, cornac_seed)
		new_performance = self.calculate_performance(k, train_data_new, test_data, use_bias, cornac_seed)

		if (old_performance is None) | (new_performance is None):
			delta_performance = None
		else:
			delta_performance = new_performance - old_performance

		return delta_performance


	def calculate_delta_performances(self, k, new_future_snapshots_dict_entries, use_bias = True, cornac_seed = 123):
		""" Calculate all delta performances, that is the difference in performance between recommenders on the training data
			entries in the <self.future_snapshots_dict> and the <new_future_snapshots_dict_entries>.
		"""
		# set old, and new train dfs as <list> of <pandas.DataFrame> ordered by self.algorithm.peers()
		train_dfs_old_dict = self.algorithm.get_future_snapshots_dict()
		train_dfs_old = [train_dfs_old_dict[peer] for peer in self.algorithm.peers()]
		train_dfs_new = new_future_snapshots_dict_entries
		test_dfs      = self.val_dfs

		# convert the dataframes into cornac format
		#
		if self.algorithm.algo_string in ["DecAggCFv2", "DecAggCFv3", "DecAggCFv6", "DecShCFv2", "DecShCFv3"]:
			train_datas_old = [convert_df_to_cornac_data(df, sender_as_userId = True) for df in train_dfs_old]
			train_datas_new = [convert_df_to_cornac_data(df, sender_as_userId = True) for df in train_dfs_new]
		elif self.algorithm.algo_string in ["DecCF"]:
			train_datas_old = [convert_df_to_cornac_data(df, sender_as_userId = False) for df in train_dfs_old]
			train_datas_new = [convert_df_to_cornac_data(df, sender_as_userId = False) for df in train_dfs_new]
		else:
			print("Specify Algo STRING in ./src/parameter_control/DistributedGradientTracking.py")

		test_datas      = [convert_df_to_cornac_data(df, sender_as_userId = False) for df in test_dfs]

		delta_performances = [self.calculate_delta_performance(k, train_data_old, train_data_new, test_data, use_bias = use_bias, cornac_seed = cornac_seed) \
																	for train_data_old, train_data_new, test_data in zip(train_datas_old, train_datas_new, test_datas)]

		return delta_performances # ordered by self.algorithm.peers()


	#
	# (II) UPDATE PARAMETERS (MAIN FUNCTIONALITY)
	#

	def update_parameters(self, t, new_future_snapshots_dict_entries):
		"""
		Implements (I) UPDATE FORMULA in the doc string.
		"""
		# Calculate delta_performance for all users
		k = 100
		# Calculate delta performances
		delta_performances = self.calculate_delta_performances(k, new_future_snapshots_dict_entries) # ordered by self.algorithm.peers()
		# Lookup children of every <peer> at epoch <self.algorithm.t> in the execution graph <self.algorithm.graph>
		# Recall that children represent users that send a payload to peer, if sufficiently similar (sim(peer,child)> self.algorithm.min_sim_to_sender_dict[peer])
		# The children_dict determines the peers that are relevant for the parameter update as per the update formula (I)
		children_dict = {peer:[el.split(":")[0] for el in self.algorithm.graph.children(peer + ":" + str(t))] for peer in self.algorithm.peers()}

		#######################################################################
		# (I.1) UPDATE the graph (time horizon T)
		#######################################################################
		# get current execution graph
		current_graph = self.algorithm.get_graph()

		# for every peer in self.algorithm.peers
		# get the current time horizons
		peer_Ts = [current_graph.T_of_peer(peer) for peer in self.algorithm.peers()] # ordered by self.algorithm.peers()
		# calculate the new time horizon T_peer
		#				   (peer1_T, peer2_T, self.alpha_T, self.beta_T, self.epsilon, delta_performances[self.algorithm.peers().index(peer)], min_param_1 = t)
		new_peer_Ts = []
		# CAVEAT Only consider integer values time horizons!
		for peer, peer_T in zip(self.algorithm.peers(), peer_Ts):
			new_peer_Ts.append(int(self.update_formula(peer_Ts[self.algorithm.peers().index(peer)], self.epsilon, delta_performances[self.algorithm.peers().index(peer)], t)))

		# augment graph according to the novel time horizons, if users find that they should 'collect more', then they might need to add edges if <peer_Ts[peer]> < <new_peer_Ts[peer]>
		# , that is if users find that they should 'collect more', then they might need to add edges if <peer_Ts[peer]> < <new_peer_Ts[peer]>
		# CAVEAT this is not 'rewiring
		# E.G. if peer_Ts = [3,3,3] and new_peer_Ts = [3,4,4]
		# then this will invoke <self.algorithm.graph.mobility_model.generate_graph_at_t> and create edges for epoch t=4 according to the mobility_model beyond the previous max. time horizon 3.
		# the novel time horizon thus becomes 4. Note that this code does not alter any edges before min(peer_Ts).
		###################################################
		# update time horizons; pruning may cause the current peer_Ts to be smaller than the new_peer_Ts
		#peer_Ts = [current_graph.T_of_peer(peer) for peer in self.algorithm.peers()] # ordered by self.algorithm.peers()
		# create peer_counter_df (the number of connections to sample for every peer at epoch t)
		# initialize empty peer_counter_df [index: userIds]
		peer_counter_df = pd.DataFrame(columns = ["counter"])
		# for every epoch <t> in  <min(peer_Ts)> to <max(new_peer_Ts)>:
		for t_for in range(min(peer_Ts),max(new_peer_Ts)+1,1):
			# for every <peer> in <peers>
			for peer in self.algorithm.peers():
				# create vertex for <peer> at epoch <t>
				self.algorithm.graph.add_vertex(peer+":"+str(t_for))

				# DIESES IF IST NIE DER FALL bei quickstop
				# if <peer_Ts[peer]> <  t_for <= new_peer_Ts[peer]: #
				if peer_Ts[self.algorithm.peers().index(peer)] < t_for <= new_peer_Ts[self.algorithm.peers().index(peer)]:
					# peer_counter_df[peer]+= self.mobility_model.N - <current_no_of_children_of_peer>
					peer_counter_df.loc[peer,"counter"] = self.algorithm.graph.mobility_model.N - len(self.algorithm.graph.children(peer+":"+str(t_for)))

			# augment graph according to the novel time horizon
			current_graph, _ = self.algorithm.graph.mobility_model.generate_graph_at_t(current_graph, t_for, peer_counter_df, self.algorithm.graph.mobility_model.random_gen)
		#"""

		# set new execution graph
		#########################
		new_graph = current_graph
		self.algorithm.set_graph(new_graph)

		# update self.algorithm.min_local_time_horizon
		self.algorithm.min_local_time_horizon = min(new_peer_Ts)

		#######################################################################
		# (I.2) UPDATE dynamic_percentile (do not change dynamic_percentile)  #
		#######################################################################
		new_dynamic_percentile_dict = self.algorithm.get_dynamic_percentile_dict()
		# set new dynamic percentiles
		self.algorithm.set_dynamic_percentile_dict(new_dynamic_percentile_dict)

		# (I.2.1) adjust min_sim_to_sender on the basis of the new dynamic percentiles
		self.algorithm.initialize_min_sim_to_sender_dict()

		return



if __name__ == "__main__":





	#
	# USE FOR WRITING TEST ON <DistributedGradientTracking.update_formula>
	#
	"""
	param_control = DistributedGradientTracking(1,2)
	old_param_1 = 4
	old_param_2 = 5
	alpha_p = 1.0
	beta_p = 1.0
	epsilon = 0.2
	delta_performance = 0.3



	new_param = param_control.update_formula(old_param_1, old_param_2, alpha_p, beta_p, epsilon, delta_performance)
	print("old param 1: {}".format(old_param_1))
	print("old param 2: {}".format(old_param_2))
	print("new param 1: {}".format(new_param))
	"""
	#
	#
	#
