
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)

from parameter_control.ParameterControlTemplate import ParameterControlTemplate

from utilities.util import convert_df_to_cornac_data

import pandas as pd
import numpy as np


import cornac
from cornac.experiment import Experiment
from cornac.eval_methods import BaseMethod
from cornac.models import MF, SVD#, PMF, BPR, UserKNN, ItemKNN, MCF, WMF, SVD, MCF, WMF, VAECF, CVAECF, CVAE, GMF, MCF, NeuMF, ConvMF, CDL, HPF, FM, GlobalAvg, MMMF, MostPop
from cornac.metrics import RMSE#, MAE, Precision, Recall, NDCG, AUC, MAP
#from cornac.hyperopt import Discrete, GridSearch

# for debugging:
from mobility_models.AssignNMobility import AssignNMobility


class DistributedGradientTrackingv4(ParameterControlTemplate):
	""" Class that defines a Distributed Gradient Tracking (DGT) parameter control mechanism. DGT differs from Distributed Gradient Descent (DGD) in that we employ fixed gradients.
		DGT has been used to address the speed-accuracy dilemma (source: Daneshmand et al. (2018) arXiv:1809.08694).

		DistributedGradientTracking.update_parameters applies the following formula to update <self.graph> (that is time horizons of peers) and <self.dynamic_percentile_dict> (dissemination parameters of peers).

		(I) UPDATE FORMULA:
		####################
																  _______________________________________________
																 |                                               |
		Delta p_u(t) = [beta_p * (p_u(t) - p_v(t))] - [alpha_p * | 1	, if   epsilon < |Delta performance(t)|  |]
																 | 0	, else epsilon >=|Delta performance(t)|  |
																 |_______________________________________________|
																 						Î -> we call this the 'gradient'

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
	def __init__(self, algorithm, val_df, alpha_dynamic_percentile, beta_dynamic_percentile, alpha_T, beta_T, epsilon, mobility_model = None):
		# call the template's initialization
		super().__init__(algorithm, val_df = val_df)
		# update parameters
		self.alpha_dynamic_percentile = alpha_dynamic_percentile
		self.beta_dynamic_percentile = beta_dynamic_percentile
		self.alpha_T     = alpha_T
		self.beta_T     = beta_T
		#
		self.epsilon = epsilon

		# parameter control string for future reference
		self.parameter_control_string	= "DGTv4"

		#
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

	def update_formula(self, param_1, params_2, alpha_p, beta_p, epsilon, delta_performance, min_param_1 = None, max_param_1 = None,\
				none_gradient = 1, zero_gradient = -1, saturation_gradient = -1, non_saturation_gradient = 0, use_consensus_term = True):
		"""
		DGT:	none_gradient = 1, zero_gradient = -1, saturation_gradient = -1, non_saturation_gradient = 0, use_consensus_term = True
		R&I:	none_gradient = 1, zero_gradient = -1, saturation_gradient = -1, non_saturation_gradient = 0, use_consensus_term = False
		R  :	none_gradient = 1, zero_gradient =  0, saturation_gradient =  0, non_saturation_gradient = 0, use_consensus_term = False

		Calculate the updated param_1 according to formula (I).


		Example:
			param_control = DistributedGradientTracking(1,2)
			old_param_1 = 4
			old_param_2 = 5
			alpha_p = 1.0
			beta_p = 1.0
			epsilon = 0.2
			delta_performance = 0.3

			Run:
				>> new_param = param_control.update_formula(old_param_1, old_param_2, alpha_p, beta_p, epsilon, delta_performance)

			Yields:
				consensus_term  =  1.0
				innovation term = -1.0
				old param 1     =  4
				old param 2     =  5
				new param 1     =  4.0

			[Option]: This function can be extended to return not only new_param_1 but also new_param_2.

			REMARK	gradient = 0 indicates that the user still needs to collect more data (do not change parameters);
					gradient = 1 indicates that the user can start decreasing data collection (change parameters).
		"""
		# consensus term
		#################
		print(params_2)
		if use_consensus_term:
			print(np.array([param_2 - param_1 for param_2 in params_2]))
			consensus_term = beta_p * ( np.array([param_2 - param_1 for param_2 in params_2]).mean() )
		else:
			consensus_term = 0

		# innovation term
		##################
		#
		# We encode the strategies with which we create the innovation term as a four-tuple
		# encoding the gradients to choose when
		# 1. performance differential == None (nothing collected so far)
		# 2. performance differential == 0 (no payload has been received in this epoch)
		# 3. |performance differential| < epsilon (performance changes marginally - data collection saturates)
		# 4. |performance differential| >= epsilon (performance changes largely - data collection does not saturate yet)
		# We then denote by DGT[1,2,3,4] a Distributed Gradient Tracking parameter control mechansism
		# in which uses
		# 		* gradient of value 1 if nothing has been collected so far
		# 		* gradient of value 2 if no payload has been received in this epoch
		#		* gradient of value 3 if the performance saturates
		# 		* gradient of value 4 if the performance does not yet saturate
		if delta_performance is None: # if no evaluation is possible, this indicates that the user still needs to collect data!
			gradient = none_gradient #0
		elif abs(delta_performance) == 0: # no change in collected data
			gradient = zero_gradient # 0
		elif (abs(delta_performance)<epsilon ): # if performance differentials are small (< epsilon); change parameters to lower collection
			gradient = saturation_gradient # 1
		else: # if performance differentials are large; do not change parameters
			gradient = non_saturation_gradient# 0

		# NEW in v4:
		# gradient dampening
		gradient = gradient / max(len(params_2),1)#len(params_2) is the number of children, that is connections that peer makes in this epoch


		innovation_term = alpha_p * gradient
		# delta_param = consensus_term + innovation_term
		delta_param_1 = consensus_term + innovation_term
		# new_param = param_1 + delta_param_1
		new_param_1 = param_1 + delta_param_1

		# check boundaries
		if max_param_1 is not None:
			new_param_1 = min(new_param_1, max_param_1)
		if min_param_1 is not None:
			new_param_1 = max(new_param_1, min_param_1)

		return new_param_1


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
		# get the current time horizons
		current_time_horizon_dict = self.algorithm.get_time_horizon_dict()
		# calculate the new time horizon T_peer
		new_time_horizon_dict = {}
		# CAVEAT Only consider integer values time horizons!
		# TODO: Makes this clean, to work with arbitrary amounts of children
		for peer in self.algorithm.peers():
			# if there are no children (no connections in this epoch): only consider the innovation term for parameter update
			if children_dict[peer] == []:
				new_time_horizon_dict[peer] = int(self.update_formula(current_time_horizon_dict[peer], [0], \
										self.alpha_T, 0, self.epsilon,  delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0))
			# OLD:
			#	new_time_horizon_dict[peer] = int(self.update_formula(current_time_horizon_dict[peer], 0, \
			#								self.alpha_T, 0, self.epsilon,  delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0))
			# if there are children (connections in this epoch): consider both consensus and innovation terms for parameter update
			else:
				print("children_dict[peer]", children_dict[peer]) #['2']
				print("[current_time_horizon_dict[child] for child in children_dict[peer]]", [current_time_horizon_dict[child] for child in children_dict[peer]])

				new_time_horizon_dict[peer] = int(self.update_formula(current_time_horizon_dict[peer], [current_time_horizon_dict[child] for child in children_dict[peer]], \
											self.alpha_T, self.beta_T, self.epsilon,  delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0))
				# OLD:
				#new_time_horizon_dict[peer] = int(self.update_formula(current_time_horizon_dict[peer], current_time_horizon_dict[children_dict[peer][0]], \
				#							self.alpha_T, self.beta_T, self.epsilon,  delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0))

		# set new time horizons
		self.algorithm.set_time_horizon_dict(new_time_horizon_dict)

		#######################################################################
		# (I.2) UPDATE dynamic_percentile (dissemination parameter theta)
		#######################################################################
		new_dynamic_percentile_dict = {}
		for peer in self.algorithm.peers():

			if children_dict[peer] == []:
				# if there are no children (no connections in this epoch): only consider the innovation term for parameter update
				# OLD:
				#new_dynamic_percentile_dict[peer] = self.update_formula(self.algorithm.dynamic_percentile_dict[peer], 0,\
				#		self.alpha_dynamic_percentile, 0, self.epsilon, delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0.0, max_param_1 = 1.0)
				new_dynamic_percentile_dict[peer] = self.update_formula(self.algorithm.dynamic_percentile_dict[peer], [0],\
						self.alpha_dynamic_percentile, 0, self.epsilon, delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0.0, max_param_1 = 1.0)
			else:
				# if there are children (connections in this epoch): consider both consensus and innovation terms for parameter update
				# OLD:
				#new_dynamic_percentile_dict[peer] = self.update_formula(self.algorithm.dynamic_percentile_dict[peer], self.algorithm.dynamic_percentile_dict[children_dict[peer][0]],\
				#		self.alpha_dynamic_percentile, self.beta_dynamic_percentile, self.epsilon, delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0.0, max_param_1 = 1.0)
				new_dynamic_percentile_dict[peer] = self.update_formula(self.algorithm.dynamic_percentile_dict[peer], [self.algorithm.dynamic_percentile_dict[child] for child in children_dict[peer]],\
						self.alpha_dynamic_percentile, self.beta_dynamic_percentile, self.epsilon, delta_performances[self.algorithm.peers().index(peer)], min_param_1 = 0.0, max_param_1 = 1.0)

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
