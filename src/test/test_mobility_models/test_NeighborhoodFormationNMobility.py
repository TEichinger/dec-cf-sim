# import testing module
import unittest

# add upper directory to sys-path to find graph.py
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[1]) # ./src
root_dir   = str(file_dir.parents[2]) # ./
sys.path.insert(0, src_dir)

import pandas as pd

from mobility_models.NeighborhoodFormationNMobility import NeighborhoodFormationNMobility
from collections import deque


class TestNeighborhoodFormationNMobility(unittest.TestCase):
	""" Test whether the UniformRandomNMobility produces the desired execution graph. We will consider the Example given in the Class's doc-string.

		Example:
		########

		- rating_df	:	Extracted from the './data/test' file; this is a list with columns ["userId", "itemId", "rating", and "timestamp"]
						It includes 8 users '1' through '8' with about 5 ratings per user
						NOTE that the rating_df is necessary for the creation of the execution graph, since exchanges only happen between mutually similar peers.
		- N 			= 2
		- sim_string	= "cosine"
		- min_sim		= 0.0

		Then the generated graph is represented by the following directed edges

		Example:
		########

		The rating_df given in the test-file (./data/test) will produce the following mobility_model.graph

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

		# 3. Test graph generation

		# 3.1 Correct vertices
		# 3.2 Correct edges
		# 3.3 Children can correctly be found
		# 3.4 History can correctly be found
		# 3.5 Test whether kwarg 'min_sim' filters all neighbor candidates <= min_sim
		# 3.6 [NOT GIVEN] Test whether the outputs due to similarity collisions are stable --> this is generally not true,
						however, I have currently not seen an instance in which reruns had always distinct outcomes.


		# This is the dictionary representation of the test-file (at ./data/test) which holds a test set of 8 synthetic users '1' through '8' on some synthetic items.
		test_dict = {'userId': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '2', 6: '2', 7: '2', 8: '2', 9: '2', 10: '2', 11: '3', 12: '3', 13: '3', 14: '3', 15: '4', \
						16: '4', 17: '4', 18: '5', 19: '5', 20: '5', 21: '5', 22: '6', 23: '6', 24: '6', 25: '6', 26: '7', 27: '7', 28: '7', 29: '7', 30: '8', 31: '8',\
						32: '8', 33: '8'}, 'itemId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 3, 7: 4, 8: 5, 9: 7, 10: 8, 11: 1, 12: 2, 13: 5, 14: 6, 15: 10, 16: 12, \
						17: 11, 18: 1, 19: 2, 20: 10, 21: 15, 22: 15, 23: 3, 24: 2, 25: 5, 26: 1, 27: 12, 28: 13, 29: 20, 30: 20, 31: 21, 32: 22, 33: 23}, \
						'rating': {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 3.0, 7: 5.0, 8: 3.0, 9: 2.0, 10: 5.0, 11: 3.0, 12: 4.0, 13: 1.0, 14: 3.0, \
						15: 4.0, 16: 5.0, 17: 3.0, 18: 2.0, 19: 3.0, 20: 5.0, 21: 1.0, 22: 1.0, 23: 5.0, 24: 5.0, 25: 5.0, 26: 2.0, 27: 3.0, 28: 2.0, 29: 4.0, 30: 1.0, 31: 5.0, 32: 2.0, 33: 2.0}}


	"""
	###############################
	# 3. Graph Generation         #
	###############################

	def test_generate_graph(self):
		""" Test wether the graph generation works correctly. """
		# SET PARAMETERS
		N = 2
		sim_string   	= "cosine"
		min_sim			= 0.0

		# LOAD DATA
		# the test_dict represents the data in ./data/test for easy loading into a pandas.DataFrame
		test_dict = {'userId': {0: '1', 1: '1', 2: '1', 3: '1', 4: '1', 5: '2', 6: '2', 7: '2', 8: '2', 9: '2', 10: '2', 11: '3', 12: '3', 13: '3', 14: '3', 15: '4', \
						16: '4', 17: '4', 18: '5', 19: '5', 20: '5', 21: '5', 22: '6', 23: '6', 24: '6', 25: '6', 26: '7', 27: '7', 28: '7', 29: '7', 30: '8', 31: '8',\
						32: '8', 33: '8'}, 'itemId': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 1, 6: 3, 7: 4, 8: 5, 9: 7, 10: 8, 11: 1, 12: 2, 13: 5, 14: 6, 15: 10, 16: 12, \
						17: 11, 18: 1, 19: 2, 20: 10, 21: 15, 22: 15, 23: 3, 24: 2, 25: 5, 26: 1, 27: 12, 28: 13, 29: 20, 30: 20, 31: 21, 32: 22, 33: 23}, \
						'rating': {0: 1.0, 1: 2.0, 2: 5.0, 3: 4.0, 4: 3.0, 5: 2.0, 6: 3.0, 7: 5.0, 8: 3.0, 9: 2.0, 10: 5.0, 11: 3.0, 12: 4.0, 13: 1.0, 14: 3.0, \
						15: 4.0, 16: 5.0, 17: 3.0, 18: 2.0, 19: 3.0, 20: 5.0, 21: 1.0, 22: 1.0, 23: 5.0, 24: 5.0, 25: 5.0, 26: 2.0, 27: 3.0, 28: 2.0, 29: 4.0, 30: 1.0, 31: 5.0, 32: 2.0, 33: 2.0}}
		rating_df = pd.DataFrame.from_dict(test_dict)
		peers = rating_df.loc[:,"userId"].drop_duplicates()


		#
		mobility_model = NeighborhoodFormationNMobility(rating_df, N, sim_string = sim_string, min_sim = min_sim)
		mobility_model.generate_graph()

		# 3.1 Correct vertices
		self.assertEqual(mobility_model.graph.vertices(), ['1:0', '2:0', '3:0', '4:0', '5:0', '6:0', '7:0', '8:0', '1:1', '2:1', '3:1', '4:1', '5:1', '6:1', '7:1', '8:1'])
		# 3.2 Correct edges
		self.assertEqual(mobility_model.graph.edges(), [('1:1', '6:0'), ('1:1', '2:0'), ('2:1', '1:0'), ('2:1', '6:0'), ('3:1', '5:0'), ('3:1', '6:0'), ('4:1', '5:0'),\
											('4:1', '7:0'), ('5:1', '3:0'), ('5:1', '4:0'), ('6:1', '1:0'), ('6:1', '3:0'), ('7:1', '4:0'), ('7:1', '3:0'), ('8:1', '7:0')])
		# 3.3 Children can correctly be found
		self.assertEqual(mobility_model.graph.children('3:1'), deque(['5:0', '6:0']))
		self.assertEqual(mobility_model.graph.children('8:1'), deque(['7:0']))
		# 3.4 History can correctly be found
		self.assertEqual(mobility_model.graph.collect_history('7:1'), ['4:0', '3:0'])
		# 3.5 Test kwarg 'min_sim'. Here we check whether setting 'min_sim' = -1.0 causes peer '8' to have N=2 neighbors instead of only 1.
		# re-instantiate the graph
		min_sim			= -1.0
		mobility_model = NeighborhoodFormationNMobility(rating_df, N, sim_string = sim_string, min_sim = min_sim)
		mobility_model.generate_graph()
		self.assertEqual(len(mobility_model.graph.children('8:1')), N)
		# 3.6 Check whether the output of neighbors with equal similarities (for instance peer '8' has similiarity 0.0 to all peers except '7'
		# we just re-instantiate again and expect reproducible results even with similarity collisions


if __name__ == '__main__':
	# Run unit tests
	unittest.main()
