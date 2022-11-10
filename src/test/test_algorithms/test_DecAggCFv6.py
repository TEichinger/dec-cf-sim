# import testing module
import unittest

# add upper directory to sys-path to find graph.py
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[1]) # ./src
templates_dir = os.path.join(src_dir, "algorithms")
root_dir   = str(file_dir.parents[2]) # ./
sys.path.insert(0, src_dir)
sys.path.insert(0, templates_dir)


# for production
import pandas as pd
import numpy as np

from algorithms.DecAggCFv6 import DecAggCFv6
from mobility_models.graph import DAG
from parameter_control.StaticParameters import StaticParameters
from utilities.util import build_execution_graph, make_sim_dict_pickle_string, make_snapshot_info_string, make_initial_snapshot_string
from collections import deque

# for debugging
from mobility_models.AssignNMobility import AssignNMobility





class TestDecAggCFv6(unittest.TestCase):
	""" Tests are conducted on the test data set ./data/test which features 8 synthetic users on some synthetic items.

		# 1. Test the Overall Functionality of the DecAggCFv6 algorithm [only depends on the chosen parameters for the algorithm and not on the mobility model (including the DAG class)]

		# 1.1 Test DecAlgoTemplate.fill_snapshots
		# 1.2 Test DecAlgoTemplate.fill_snapshots_inparallel



	"""
	#################################################################
	# 1. Test the Overall Functionality of the DecAggCFv6 algorithm #
	#################################################################

	def test_overall_DecAggCFv6(self):
		""" Doc-string

		# This is the dictionary representation of the test-file (at ./data/test) which holds a test set of 8 synthetic users '1' through '8' on some synthetic items.
		test_dict = {'userId': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '2', 6: '2', 7: '2', 8: '2', 9: '2', 10: '2', 11: '3', 12: '3', 13: '3', 14: '3', 15: '4', \
						16: '4', 17: '4', 18: '5', 19: '5', 20: '5', 21: '5', 22: '6', 23: '6', 24: '6', 25: '6', 26: '7', 27: '7', 28: '7', 29: '7', 30: '8', 31: '8',\
						32: '8', 33: '8'}, 'itemId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 3, 7: 4, 8: 5, 9: 7, 10: 8, 11: 1, 12: 2, 13: 5, 14: 6, 15: 10, 16: 12, \
						17: 11, 18: 1, 19: 2, 20: 10, 21: 15, 22: 15, 23: 3, 24: 2, 25: 5, 26: 1, 27: 12, 28: 13, 29: 20, 30: 20, 31: 21, 32: 22, 33: 23}, \
						'rating': {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 3.0, 7: 5.0, 8: 3.0, 9: 2.0, 10: 5.0, 11: 3.0, 12: 4.0, 13: 1.0, 14: 3.0, \
						15: 4.0, 16: 5.0, 17: 3.0, 18: 2.0, 19: 3.0, 20: 5.0, 21: 1.0, 22: 1.0, 23: 5.0, 24: 5.0, 25: 5.0, 26: 2.0, 27: 3.0, 28: 2.0, 29: 4.0, 30: 1.0, 31: 5.0, 32: 2.0, 33: 2.0}}

		"""
		# DATA
		# the test_dict represents the data in ./data/test for easy loading into a pandas.DataFrame
		test_dict = {'userId': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '2', 6: '2', 7: '2', 8: '2', 9: '2', 10: '2', 11: '3', 12: '3', 13: '3', 14: '3', 15: '4', \
						16: '4', 17: '4', 18: '5', 19: '5', 20: '5', 21: '5', 22: '6', 23: '6', 24: '6', 25: '6', 26: '7', 27: '7', 28: '7', 29: '7', 30: '8', 31: '8',\
						32: '8', 33: '8'}, 'itemId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 3, 7: 4, 8: 5, 9: 7, 10: 8, 11: 1, 12: 2, 13: 5, 14: 6, 15: 10, 16: 12, \
						17: 11, 18: 1, 19: 2, 20: 10, 21: 15, 22: 15, 23: 3, 24: 2, 25: 5, 26: 1, 27: 12, 28: 13, 29: 20, 30: 20, 31: 21, 32: 22, 33: 23}, \
						'rating': {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 3.0, 7: 5.0, 8: 3.0, 9: 2.0, 10: 5.0, 11: 3.0, 12: 4.0, 13: 1.0, 14: 3.0, \
						15: 4.0, 16: 5.0, 17: 3.0, 18: 2.0, 19: 3.0, 20: 5.0, 21: 1.0, 22: 1.0, 23: 5.0, 24: 5.0, 25: 5.0, 26: 2.0, 27: 3.0, 28: 2.0, 29: 4.0, 30: 1.0, 31: 5.0, 32: 2.0, 33: 2.0}}
		rating_df = pd.DataFrame.from_dict(test_dict)
		peers = rating_df.loc[:,"userId"].drop_duplicates()

		#############################################################################
		# graph_dict is the execution graph obtained by the following procedure     #
		#                                                                           #
		#N = 3                                                                      #
		#T = 5                                                                      #
		#random_seed = 1234                                                         #
		#graph			= DAG(peers=peers)                                          #
		#mobility_model	= AssignNMobility(graph, N, T, random_seed=random_seed)     #
		#execution_graph = build_execution_graph(peers, mobility_model)             #
		#############################################################################

		# DATA
		graph_dict   = {'1:0': deque([]), '2:0': deque([]), '3:0': deque([]), '4:0': deque([]), '5:0': deque([]), '6:0': deque([]), '7:0': deque([]), '8:0': deque([]), '1:1': deque(['8:0', '4:0', '2:0']), '8:1': deque(['1:0', '2:0', '5:0']), '4:1': deque(['1:0', '3:0', '5:0']), '2:1': deque(['1:0', '8:0', '7:0']), '7:1': deque(['2:0', '3:0', '6:0']), '3:1': deque(['6:0', '7:0', '4:0']), '6:1': deque(['3:0', '5:0', '7:0']), '5:1': deque(['4:0', '6:0', '8:0']), '1:2': deque(['3:1', '2:1', '7:1']), '3:2': deque(['1:1', '7:1', '4:1']), '2:2': deque(['1:1', '5:1', '7:1']), '7:2': deque(['1:1', '2:1', '3:1']), '5:2': deque(['2:1', '6:1', '8:1']), '4:2': deque(['3:1', '6:1', '8:1']), '6:2': deque(['4:1', '5:1', '8:1']), '8:2': deque(['4:1', '5:1', '6:1']), '1:3': deque(['3:2', '2:2', '4:2']), '3:3': deque(['1:2', '4:2', '7:2']), '2:3': deque(['1:2', '5:2', '8:2']), '4:3': deque(['1:2', '3:2', '6:2']), '5:3': deque(['2:2', '6:2', '7:2']), '8:3': deque(['2:2', '6:2', '7:2']), '7:3': deque(['3:2', '5:2', '8:2']), '6:3': deque(['4:2', '5:2', '8:2']), '1:4': deque(['5:3', '6:3', '2:3']), '5:4': deque(['1:3', '3:3', '8:3']), '6:4': deque(['1:3', '2:3', '3:3']), '2:4': deque(['1:3', '6:3', '4:3']), '4:4': deque(['2:3', '8:3', '7:3']), '3:4': deque(['6:3', '7:3', '5:3']), '7:4': deque(['3:3', '4:3', '8:3']), '8:4': deque(['4:3', '5:3', '7:3']), '1:5': deque(['4:4', '3:4', '8:4']), '4:5': deque(['1:4', '7:4', '5:4']), '3:5': deque(['1:4', '2:4', '6:4']), '8:5': deque(['1:4', '6:4', '7:4']), '2:5': deque(['5:4', '6:4', '3:4']), '5:5': deque(['2:4', '4:4', '7:4']), '6:5': deque(['2:4', '3:4', '8:4']), '7:5': deque(['4:4', '5:4', '8:4'])}
		execution_graph = DAG(graph_dict = graph_dict)


		dataset_string	= "test"
		n_splits     	= None
		random_state 	= None # for train/test
		topN         	= 3	  # for collect_payload
		random_seed		= 1234
		N				= 3
		T            	= 5
		sim_string   	= "cosine"
		foldNr			= None
		output_dir		= os.path.join(root_dir,"data/snapshots")
		mobility_model	= AssignNMobility
		mobility_string = "AssignNMobility"

		sim_mat_path	= None

		min_sim_to_child_child  = 0.0
		min_sim_to_sender_dynamic = True
		percentile		= 0.70
		min_sim_to_sender		= 0.0
		max_agg_sim				= 100
		hide_seed				= 104
		hide_p					= None
		timestamp_delta			= None

		save_every_i			= 10

		sim_dict_pickle_string = make_sim_dict_pickle_string(dataset_string, random_state, random_seed, hide_seed, hide_p, T, sim_string)
		snapshot_info_string = make_snapshot_info_string(topN, sim_string, dataset_string, random_seed, n_splits, random_state, N, \
									mobility_string, percentile, hide_p = hide_p, timestamp_delta = timestamp_delta)
		initial_snapshot_string = make_initial_snapshot_string(dataset_string, random_state)

		algorithm = DecAggCFv6(execution_graph, rating_df, snapshot_info_string = snapshot_info_string, dataset_string = dataset_string, initial_snapshot_string = initial_snapshot_string, output_dir = output_dir, sim_string = sim_string,
						topN = topN, min_sim_to_child_child	= min_sim_to_child_child, min_sim_to_sender	= min_sim_to_sender, max_agg_sim = max_agg_sim, sim_mat_path = sim_mat_path,\
						min_sim_to_sender_dynamic = min_sim_to_sender_dynamic, dynamic_percentile = percentile, hide_seed = hide_seed, hide_p = hide_p, sim_dict_pickle_string = sim_dict_pickle_string,\
						timestamp_delta = timestamp_delta, save_every_i = save_every_i)

		# use static parameters
		parameter_control_model = StaticParameters(algorithm)
		algorithm.use_parameter_control_model(parameter_control_model)

		# DATA
		# We save the result of the fill_snapshots method and evaluate the result on the future_snapshots_dict, that is the databases that every peer holds after T periods
		# NOTE that we copied the dictionaries from the shell and replaced np.np.np.np.np.nan with np.np.np.np.np.np.nan
		true_future_snapshots_dict = dict()

		true_future_snapshots_dict["1"] = {'userId': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '1', 6: '1', 7: '1', 8: '1', 9: '1', 10: '1', 11: '1', 12: '1', 13: '1', 14: '1', 15: '1', 16: '1', 17: '1', 18: '1', 19: '1', 20: '1', 21: '1', 22: '1', 23: '1', 24: '1', 25: '1', 26: '1', 27: '1', 28: '1', 29: '1', 30: '1', 31: '1', 32: '1'}, 'itemId': {0: 1, 1: 3, 2: 4, 3: 5, 4: 7, 5: 8, 6: 1, 7: 2, 8: 3, 9: 4, 10: 5, 11: 7, 12: 8, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 7, 19: 8, 20: 1, 21: 2, 22: 3, 23: 5, 24: 6, 25: 15, 26: 1, 27: 2, 28: 3, 29: 4, 30: 5, 31: 7, 32: 8}, 'rating': {0: 2.0, 1: 3.0000000000000004, 2: 5.0, 3: 3.0000000000000004, 4: 2.0, 5: 5.0, 6: 1.415714291203977, 7: 2.0, 8: 4.168571417592046, 9: 4.415714291203977, 10: 3.0000000000000004, 11: 2.0, 12: 5.0, 13: 1.477883551245696, 14: 2.0, 15: 4.044232897508609, 16: 4.4778835512456965, 17: 3.0000000000000004, 18: 2.0, 19: 4.999999999999999, 20: 3.0000000000000004, 21: 4.707913730882081, 22: 5.0, 23: 3.831654923528326, 24: 3.0000000000000004, 25: 1.0, 26: 1.489801374040583, 27: 2.0, 28: 4.020397251918835, 29: 4.489801374040583, 30: 3.0000000000000004, 31: 2.0, 32: 5.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: np.nan, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan, 31: np.nan, 32: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: np.nan, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan, 31: np.nan, 32: np.nan}, 'sender': {0: '2', 1: '2', 2: '2', 3: '2', 4: '2', 5: '2', 6: '2', 7: '2', 8: '2', 9: '2', 10: '2', 11: '2', 12: '2', 13: '2', 14: '2', 15: '2', 16: '2', 17: '2', 18: '2', 19: '2', 20: '6', 21: '6', 22: '6', 23: '6', 24: '6', 25: '6', 26: '2', 27: '2', 28: '2', 29: '2', 30: '2', 31: '2', 32: '2'}, 'sim_to_sender': {0: 0.7114914586232077, 1: 0.7114914586232077, 2: 0.7114914586232077, 3: 0.7114914586232077, 4: 0.7114914586232077, 5: 0.7114914586232077, 6: 0.7114914586232077, 7: 0.7114914586232077, 8: 0.7114914586232077, 9: 0.7114914586232077, 10: 0.7114914586232077, 11: 0.7114914586232077, 12: 0.7114914586232077, 13: 0.7114914586232077, 14: 0.7114914586232077, 15: 0.7114914586232077, 16: 0.7114914586232077, 17: 0.7114914586232077, 18: 0.7114914586232077, 19: 0.7114914586232077, 20: 0.7733602811121824, 21: 0.7733602811121824, 22: 0.7733602811121824, 23: 0.7733602811121824, 24: 0.7733602811121824, 25: 0.7733602811121824, 26: 0.7114914586232077, 27: 0.7114914586232077, 28: 0.7114914586232077, 29: 0.7114914586232077, 30: 0.7114914586232077, 31: 0.7114914586232077, 32: 0.7114914586232077}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 3, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4, 29: 4, 30: 4, 31: 4, 32: 4}}

		true_future_snapshots_dict["2"] = {'userId': {0: '2', 1: '2', 2: '2', 3: '2', 4: '2', 5: '2', 6: '2', 7: '2', 8: '2', 9: '2', 10: '2', 11: '2', 12: '2', 13: '2', 14: '2', 15: '2', 16: '2', 17: '2', 18: '2', 19: '2', 20: '2', 21: '2', 22: '2', 23: '2', 24: '2', 25: '2', 26: '2', 27: '2', 28: '2', 29: '2', 30: '2', 31: '2', 32: '2', 33: '2', 34: '2', 35: '2', 36: '2', 37: '2', 38: '2', 39: '2'}, 'itemId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 7, 11: 8, 12: 1, 13: 2, 14: 3, 15: 4, 16: 5, 17: 7, 18: 8, 19: 1, 20: 2, 21: 3, 22: 4, 23: 5, 24: 7, 25: 8, 26: 1, 27: 2, 28: 3, 29: 5, 30: 6, 31: 15, 32: 1, 33: 2, 34: 3, 35: 4, 36: 5, 37: 7, 38: 8, 39: 15}, 'rating': {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0000000000000004, 5: 1.5842857087960227, 6: 2.0, 7: 3.8314285824079546, 8: 4.584285708796023, 9: 3.0000000000000004, 10: 2.0, 11: 5.0, 12: 1.5221164487543044, 13: 2.0, 14: 3.9557671024913925, 15: 4.522116448754304, 16: 3.0000000000000004, 17: 2.0, 18: 5.0, 19: 1.5101986259594171, 20: 2.0, 21: 3.979602748081166, 22: 4.510198625959417, 23: 3.0000000000000004, 24: 2.0, 25: 5.0, 26: 3.0000000000000004, 27: 4.693446152348254, 28: 5.0, 29: 3.773784609393018, 30: 3.0000000000000004, 31: 1.0, 32: 1.3818369326461395, 33: 3.070493790057591, 34: 4.508828844849274, 35: 4.381836932646139, 36: 3.7136625267050607, 37: 2.0, 38: 4.999999999999999, 39: 1.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: np.nan, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan, 31: np.nan, 32: np.nan, 33: np.nan, 34: np.nan, 35: np.nan, 36: np.nan, 37: np.nan, 38: np.nan, 39: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: np.nan, 25: np.nan, 26: np.nan, 27: np.nan, 28: np.nan, 29: np.nan, 30: np.nan, 31: np.nan, 32: np.nan, 33: np.nan, 34: np.nan, 35: np.nan, 36: np.nan, 37: np.nan, 38: np.nan, 39: np.nan}, 'sender': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '1', 6: '1', 7: '1', 8: '1', 9: '1', 10: '1', 11: '1', 12: '1', 13: '1', 14: '1', 15: '1', 16: '1', 17: '1', 18: '1', 19: '1', 20: '1', 21: '1', 22: '1', 23: '1', 24: '1', 25: '1', 26: '6', 27: '6', 28: '6', 29: '6', 30: '6', 31: '6', 32: '6', 33: '6', 34: '6', 35: '6', 36: '6', 37: '6', 38: '6', 39: '6'}, 'sim_to_sender': {0: 0.7114914586232077, 1: 0.7114914586232077, 2: 0.7114914586232077, 3: 0.7114914586232077, 4: 0.7114914586232077, 5: 0.7114914586232077, 6: 0.7114914586232077, 7: 0.7114914586232077, 8: 0.7114914586232077, 9: 0.7114914586232077, 10: 0.7114914586232077, 11: 0.7114914586232077, 12: 0.7114914586232077, 13: 0.7114914586232077, 14: 0.7114914586232077, 15: 0.7114914586232077, 16: 0.7114914586232077, 17: 0.7114914586232077, 18: 0.7114914586232077, 19: 0.7114914586232077, 20: 0.7114914586232077, 21: 0.7114914586232077, 22: 0.7114914586232077, 23: 0.7114914586232077, 24: 0.7114914586232077, 25: 0.7114914586232077, 26: 0.39473684210526305, 27: 0.39473684210526305, 28: 0.39473684210526305, 29: 0.39473684210526305, 30: 0.39473684210526305, 31: 0.39473684210526305, 32: 0.39473684210526305, 33: 0.39473684210526305, 34: 0.39473684210526305, 35: 0.39473684210526305, 36: 0.39473684210526305, 37: 0.39473684210526305, 38: 0.39473684210526305, 39: 0.39473684210526305}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3, 19: 4, 20: 4, 21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4, 27: 4, 28: 4, 29: 4, 30: 4, 31: 4, 32: 5, 33: 5, 34: 5, 35: 5, 36: 5, 37: 5, 38: 5, 39: 5}}


		true_future_snapshots_dict["3"] = {'userId': {0: '3', 1: '3', 2: '3', 3: '3', 4: '3', 5: '3', 6: '3', 7: '3', 8: '3', 9: '3', 10: '3', 11: '3', 12: '3', 13: '3', 14: '3', 15: '3', 16: '3', 17: '3', 18: '3', 19: '3'}, 'itemId': {0: 2, 1: 3, 2: 5, 3: 15, 4: 1, 5: 2, 6: 3, 7: 5, 8: 6, 9: 15, 10: 1, 11: 2, 12: 10, 13: 15, 14: 1, 15: 2, 16: 3, 17: 5, 18: 6, 19: 15}, 'rating': {0: 5.0, 1: 5.0, 2: 5.0, 3: 1.0, 4: 3.0000000000000004, 5: 4.326476509003815, 6: 5.0, 7: 2.305906036015258, 8: 3.0000000000000004, 9: 1.0, 10: 2.0, 11: 2.9999999999999996, 12: 5.0, 13: 1.0, 14: 3.0000000000000004, 15: 4.466148462286079, 16: 5.0, 17: 2.8645938491443177, 18: 3.0000000000000004, 19: 1.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan}, 'sender': {0: '6', 1: '6', 2: '6', 3: '6', 4: '6', 5: '6', 6: '6', 7: '6', 8: '6', 9: '6', 10: '5', 11: '5', 12: '5', 13: '5', 14: '6', 15: '6', 16: '6', 17: '6', 18: '6', 19: '6'}, 'sim_to_sender': {0: 0.48472920895592575, 1: 0.48472920895592575, 2: 0.48472920895592575, 3: 0.48472920895592575, 4: 0.48472920895592575, 5: 0.48472920895592575, 6: 0.48472920895592575, 7: 0.48472920895592575, 8: 0.48472920895592575, 9: 0.48472920895592575, 10: 0.48719876576469, 11: 0.48719876576469, 12: 0.48719876576469, 13: 0.48719876576469, 14: 0.48472920895592575, 15: 0.48472920895592575, 16: 0.48472920895592575, 17: 0.48472920895592575, 18: 0.48472920895592575, 19: 0.48472920895592575}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, 19: 5}}


		true_future_snapshots_dict["4"] = {'userId': {0: '4', 1: '4', 2: '4', 3: '4', 4: '4', 5: '4', 6: '4', 7: '4', 8: '4', 9: '4', 10: '4', 11: '4', 12: '4', 13: '4', 14: '4', 15: '4', 16: '4', 17: '4', 18: '4', 19: '4'}, 'itemId': {0: 1, 1: 2, 2: 10, 3: 15, 4: 1, 5: 12, 6: 13, 7: 20, 8: 1, 9: 10, 10: 11, 11: 12, 12: 13, 13: 20, 14: 1, 15: 2, 16: 10, 17: 11, 18: 12, 19: 15}, 'rating': {0: 2.0, 1: 3.0, 2: 5.0, 3: 1.0, 4: 2.0, 5: 3.0, 6: 2.0, 7: 4.0, 8: 2.0, 9: 4.0, 10: 3.0, 11: 4.460627536354109, 12: 2.0, 13: 4.0, 14: 2.0, 15: 3.0, 16: 4.311726507505022, 17: 3.0, 18: 5.0, 19: 1.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan}, 'sender': {0: '5', 1: '5', 2: '5', 3: '5', 4: '7', 5: '7', 6: '7', 7: '7', 8: '7', 9: '7', 10: '7', 11: '7', 12: '7', 13: '7', 14: '5', 15: '5', 16: '5', 17: '5', 18: '5', 19: '5'}, 'sim_to_sender': {0: 0.4529108136578383, 1: 0.4529108136578383, 2: 0.4529108136578383, 3: 0.4529108136578383, 4: 0.36927447293799814, 5: 0.36927447293799814, 6: 0.36927447293799814, 7: 0.36927447293799814, 8: 0.36927447293799814, 9: 0.36927447293799814, 10: 0.36927447293799814, 11: 0.36927447293799814, 12: 0.36927447293799814, 13: 0.36927447293799814, 14: 0.4529108136578383, 15: 0.4529108136578383, 16: 0.4529108136578383, 17: 0.4529108136578383, 18: 0.4529108136578383, 19: 0.4529108136578383}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5, 19: 5}}


		true_future_snapshots_dict["5"] = {'userId': {0: '5', 1: '5', 2: '5', 3: '5', 4: '5', 5: '5', 6: '5', 7: '5', 8: '5', 9: '5', 10: '5', 11: '5', 12: '5', 13: '5', 14: '5'}, 'itemId': {0: 10, 1: 11, 2: 12, 3: 1, 4: 2, 5: 3, 6: 5, 7: 6, 8: 15, 9: 1, 10: 2, 11: 10, 12: 11, 13: 12, 14: 15}, 'rating': {0: 4.0, 1: 3.0, 2: 5.0, 3: 2.9999999999999996, 4: 4.376254730329305, 5: 5.0, 6: 2.5050189213172183, 7: 2.9999999999999996, 8: 1.0, 9: 2.0, 10: 3.0, 11: 4.688273492494978, 12: 3.0, 13: 5.0, 14: 1.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan}, 'sender': {0: '4', 1: '4', 2: '4', 3: '3', 4: '3', 5: '3', 6: '3', 7: '3', 8: '3', 9: '4', 10: '4', 11: '4', 12: '4', 13: '4', 14: '4'}, 'sim_to_sender': {0: 0.4529108136578383, 1: 0.4529108136578383, 2: 0.4529108136578383, 3: 0.48719876576469, 4: 0.48719876576469, 5: 0.48719876576469, 6: 0.48719876576469, 7: 0.48719876576469, 8: 0.48719876576469, 9: 0.4529108136578383, 10: 0.4529108136578383, 11: 0.4529108136578383, 12: 0.4529108136578383, 13: 0.4529108136578383, 14: 0.4529108136578383}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 5, 10: 5, 11: 5, 12: 5, 13: 5, 14: 5}}

		true_future_snapshots_dict["6"] = {'userId': {0: '6', 1: '6', 2: '6', 3: '6', 4: '6', 5: '6', 6: '6', 7: '6', 8: '6', 9: '6', 10: '6', 11: '6', 12: '6', 13: '6', 14: '6', 15: '6', 16: '6', 17: '6', 18: '6', 19: '6', 20: '6', 21: '6', 22: '6'}, 'itemId': {0: 1, 1: 2, 2: 5, 3: 6, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 7, 10: 8, 11: 1, 12: 2, 13: 3, 14: 5, 15: 6, 16: 15, 17: 1, 18: 2, 19: 3, 20: 5, 21: 6, 22: 15}, 'rating': {0: 3.0000000000000004, 1: 4.0, 2: 1.0, 3: 3.0000000000000004, 4: 1.3818369326461395, 5: 1.9999999999999998, 6: 4.236326134707721, 7: 4.381836932646139, 8: 3.0000000000000004, 9: 2.0, 10: 4.999999999999999, 11: 3.0000000000000004, 12: 4.673523490996185, 13: 5.0, 14: 3.6940939639847423, 15: 3.0000000000000004, 16: 1.0, 17: 3.000000000000001, 18: 4.533851537713921, 19: 5.0, 20: 3.1354061508556827, 21: 3.000000000000001, 22: 1.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan}, 'sender': {0: '3', 1: '3', 2: '3', 3: '3', 4: '1', 5: '1', 6: '1', 7: '1', 8: '1', 9: '1', 10: '1', 11: '3', 12: '3', 13: '3', 14: '3', 15: '3', 16: '3', 17: '3', 18: '3', 19: '3', 20: '3', 21: '3', 22: '3'}, 'sim_to_sender': {0: 0.48472920895592575, 1: 0.48472920895592575, 2: 0.48472920895592575, 3: 0.48472920895592575, 4: 0.7733602811121824, 5: 0.7733602811121824, 6: 0.7733602811121824, 7: 0.7733602811121824, 8: 0.7733602811121824, 9: 0.7733602811121824, 10: 0.7733602811121824, 11: 0.48472920895592575, 12: 0.48472920895592575, 13: 0.48472920895592575, 14: 0.48472920895592575, 15: 0.48472920895592575, 16: 0.48472920895592575, 17: 0.48472920895592575, 18: 0.48472920895592575, 19: 0.48472920895592575, 20: 0.48472920895592575, 21: 0.48472920895592575, 22: 0.48472920895592575}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 5, 18: 5, 19: 5, 20: 5, 21: 5, 22: 5}}

		true_future_snapshots_dict["7"] = {'userId': {0: '7', 1: '7', 2: '7', 3: '7', 4: '7', 5: '7', 6: '7', 7: '7', 8: '7', 9: '7', 10: '7', 11: '7', 12: '7', 13: '7', 14: '7', 15: '7', 16: '7', 17: '7', 18: '7', 19: '7', 20: '7', 21: '7', 22: '7', 23: '7', 24: '7'}, 'itemId': {0: 1, 1: 2, 2: 5, 3: 6, 4: 1, 5: 2, 6: 5, 7: 6, 8: 1, 9: 2, 10: 5, 11: 6, 12: 1, 13: 2, 14: 5, 15: 6, 16: 10, 17: 11, 18: 12, 19: 1, 20: 10, 21: 11, 22: 12, 23: 13, 24: 20}, 'rating': {0: 2.9999999999999996, 1: 4.0, 2: 1.0, 3: 2.9999999999999996, 4: 2.9999999999999996, 5: 4.0, 6: 1.0, 7: 2.9999999999999996, 8: 2.9999999999999996, 9: 4.0, 10: 1.0, 11: 2.9999999999999996, 12: 2.9999999999999996, 13: 4.0, 14: 1.0, 15: 2.9999999999999996, 16: 4.0, 17: 3.0, 18: 5.0, 19: 2.0, 20: 4.0, 21: 3.0, 22: 3.5393724636458903, 23: 2.0, 24: 4.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan, 12: np.nan, 13: np.nan, 14: np.nan, 15: np.nan, 16: np.nan, 17: np.nan, 18: np.nan, 19: np.nan, 20: np.nan, 21: np.nan, 22: np.nan, 23: np.nan, 24: np.nan}, 'sender': {0: '3', 1: '3', 2: '3', 3: '3', 4: '3', 5: '3', 6: '3', 7: '3', 8: '3', 9: '3', 10: '3', 11: '3', 12: '3', 13: '3', 14: '3', 15: '3', 16: '4', 17: '4', 18: '4', 19: '4', 20: '4', 21: '4', 22: '4', 23: '4', 24: '4'}, 'sim_to_sender': {0: 0.17654696590094993, 1: 0.17654696590094993, 2: 0.17654696590094993, 3: 0.17654696590094993, 4: 0.17654696590094993, 5: 0.17654696590094993, 6: 0.17654696590094993, 7: 0.17654696590094993, 8: 0.17654696590094993, 9: 0.17654696590094993, 10: 0.17654696590094993, 11: 0.17654696590094993, 12: 0.17654696590094993, 13: 0.17654696590094993, 14: 0.17654696590094993, 15: 0.17654696590094993, 16: 0.36927447293799814, 17: 0.36927447293799814, 18: 0.36927447293799814, 19: 0.36927447293799814, 20: 0.36927447293799814, 21: 0.36927447293799814, 22: 0.36927447293799814, 23: 0.36927447293799814, 24: 0.36927447293799814}, 'timestamp': {0: 1, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 3, 11: 3, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 5, 20: 5, 21: 5, 22: 5, 23: 5, 24: 5}}


		true_future_snapshots_dict["8"] = {'userId': {0: '8', 1: '8', 2: '8', 3: '8', 4: '8', 5: '8', 6: '8', 7: '8', 8: '8', 9: '8', 10: '8', 11: '8'}, 'itemId': {0: 1, 1: 12, 2: 13, 3: 20, 4: 1, 5: 12, 6: 13, 7: 20, 8: 1, 9: 12, 10: 13, 11: 20}, 'rating': {0: 2.0, 1: 3.0, 2: 2.0, 3: 4.0, 4: 2.0, 5: 3.0, 6: 2.0, 7: 4.0, 8: 2.0, 9: 3.0, 10: 2.0, 11: 4.0}, 'sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan}, 'agg_sim': {0: np.nan, 1: np.nan, 2: np.nan, 3: np.nan, 4: np.nan, 5: np.nan, 6: np.nan, 7: np.nan, 8: np.nan, 9: np.nan, 10: np.nan, 11: np.nan}, 'sender': {0: '7', 1: '7', 2: '7', 3: '7', 4: '7', 5: '7', 6: '7', 7: '7', 8: '7', 9: '7', 10: '7', 11: '7'}, 'sim_to_sender': {0: 0.11941628680530642, 1: 0.11941628680530642, 2: 0.11941628680530642, 3: 0.11941628680530642, 4: 0.11941628680530642, 5: 0.11941628680530642, 6: 0.11941628680530642, 7: 0.11941628680530642, 8: 0.11941628680530642, 9: 0.11941628680530642, 10: 0.11941628680530642, 11: 0.11941628680530642}, 'timestamp': {0: 3, 1: 3, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4, 7: 4, 8: 5, 9: 5, 10: 5, 11: 5}}


		true_future_snapshots_df_1 = pd.DataFrame.from_dict(true_future_snapshots_dict["1"])
		true_future_snapshots_df_2 = pd.DataFrame.from_dict(true_future_snapshots_dict["2"])
		true_future_snapshots_df_3 = pd.DataFrame.from_dict(true_future_snapshots_dict["3"])
		true_future_snapshots_df_4 = pd.DataFrame.from_dict(true_future_snapshots_dict["4"])
		true_future_snapshots_df_5 = pd.DataFrame.from_dict(true_future_snapshots_dict["5"])
		true_future_snapshots_df_6 = pd.DataFrame.from_dict(true_future_snapshots_dict["6"])
		true_future_snapshots_df_7 = pd.DataFrame.from_dict(true_future_snapshots_dict["7"])
		true_future_snapshots_df_8 = pd.DataFrame.from_dict(true_future_snapshots_dict["8"])


		# TEST
		# 1.1 Test DecAlgoTemplate.fill_snapshots
		algorithm.fill_snapshots()


		# transformed pandas.DataFrame of the users' snapshots respectively
		# NOTE	that the reason why we are casting between pd.DataFrame and dict is the np.np.nan! type conversion causes funny issues which result in non-equality
		#		since we use the pd.DataFrame.equals method that disregards np.np.nan values such that True is still yielded with elementwise identical np.np.nan's, we
		#		have to convert to matching dtypes, for else the method fails
		future_snapshots_df_1      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["1"].to_dict())
		future_snapshots_df_2      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["2"].to_dict())
		future_snapshots_df_3      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["3"].to_dict())
		future_snapshots_df_4      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["4"].to_dict())
		future_snapshots_df_5      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["5"].to_dict())
		future_snapshots_df_6      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["6"].to_dict())
		future_snapshots_df_7      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["7"].to_dict())
		future_snapshots_df_8      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["8"].to_dict())

		# assert the equality of all future_snapshots
		self.assertTrue(future_snapshots_df_1.equals(true_future_snapshots_df_1))
		self.assertTrue(future_snapshots_df_2.equals(true_future_snapshots_df_2))
		self.assertTrue(future_snapshots_df_3.equals(true_future_snapshots_df_3))
		self.assertTrue(future_snapshots_df_4.equals(true_future_snapshots_df_4))
		self.assertTrue(future_snapshots_df_5.equals(true_future_snapshots_df_5))
		self.assertTrue(future_snapshots_df_6.equals(true_future_snapshots_df_6))
		self.assertTrue(future_snapshots_df_7.equals(true_future_snapshots_df_7))
		self.assertTrue(future_snapshots_df_8.equals(true_future_snapshots_df_8))


		# re-initialize
		processes = 2
		algorithm = DecAggCFv6(execution_graph, rating_df, snapshot_info_string = snapshot_info_string, dataset_string = dataset_string, initial_snapshot_string = initial_snapshot_string, output_dir = output_dir, sim_string = sim_string,
						topN = topN, min_sim_to_child_child	= min_sim_to_child_child, min_sim_to_sender	= min_sim_to_sender, max_agg_sim = max_agg_sim, sim_mat_path = sim_mat_path,\
						min_sim_to_sender_dynamic = min_sim_to_sender_dynamic, dynamic_percentile = percentile, hide_seed = hide_seed, hide_p = hide_p, sim_dict_pickle_string = sim_dict_pickle_string,\
						timestamp_delta = timestamp_delta, save_every_i = save_every_i, processes = processes)

		# use static parameters
		parameter_control_model = StaticParameters(algorithm)
		algorithm.use_parameter_control_model(parameter_control_model)

		# TEST
		# 1.2 Test DecAlgoTemplate.fill_snapshots_inparallel
		algorithm.fill_snapshots()

		future_snapshots_df_1      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["1"].to_dict())
		future_snapshots_df_2      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["2"].to_dict())
		future_snapshots_df_3      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["3"].to_dict())
		future_snapshots_df_4      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["4"].to_dict())
		future_snapshots_df_5      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["5"].to_dict())
		future_snapshots_df_6      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["6"].to_dict())
		future_snapshots_df_7      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["7"].to_dict())
		future_snapshots_df_8      = pd.DataFrame.from_dict(algorithm.future_snapshots_dict["8"].to_dict())


		# assert the equality of all future_snapshots
		self.assertTrue(future_snapshots_df_1.equals(true_future_snapshots_df_1))
		self.assertTrue(future_snapshots_df_2.equals(true_future_snapshots_df_2))
		self.assertTrue(future_snapshots_df_3.equals(true_future_snapshots_df_3))
		self.assertTrue(future_snapshots_df_4.equals(true_future_snapshots_df_4))
		self.assertTrue(future_snapshots_df_5.equals(true_future_snapshots_df_5))
		self.assertTrue(future_snapshots_df_6.equals(true_future_snapshots_df_6))
		self.assertTrue(future_snapshots_df_7.equals(true_future_snapshots_df_7))
		self.assertTrue(future_snapshots_df_8.equals(true_future_snapshots_df_8))


if __name__ == '__main__':
	# Run unit tests
	unittest.main()
