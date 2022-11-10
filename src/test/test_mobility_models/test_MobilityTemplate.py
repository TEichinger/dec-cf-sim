# import testing module
import unittest

# add upper directory to sys-path to find graph.py
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[1]) # ./src
root_dir   = str(file_dir.parents[2]) # ./
sys.path.insert(0, src_dir)


from mobility_models.MobilityTemplate import MobilityTemplate

import math
import random



class TestMobilityTemplate(unittest.TestCase):
	""" Test the following functionalities of the DAG (directed acyclic graph) class in the graph.py file
		0.

		1.1 Test the sample1 method
		1.1.1 Test correct sample output
		1.1.2 Test state transfer of the random generator --> do not sample the same over and over again
		1.1.3 Test whether the remainder_peers output is correct

		1.2 Test the sampleN method
		1.2.1 Test correct sample output
		1.2.2 Test state transfer of the random generator --> do not sample the same over and over again
		1.2.3 Test that kwarg 'must_sample_list' works correctly, if the must_sample list <= N
		1.2.4 Test that kwarg 'must_sample_list' works correctly, if the must_sample list > N
		1.2.5 Test that kwarg 'must_sample_list' is included in the random_sample even if the elements in 'must_sample_list' are not in 'peers'
		1.2.6 Test that kwarg 'drop_duplicates' drops all duplicates before prompting the 'random_sample'

	"""

	###############################
	# 1. Test Auxiliary Functions #
	###############################

	def test_auxiliary_functions(self):
		""" Test wether the auxiliary functions work correctly. """
		graph = None
		N = 3
		T = 2
		graph_seed = 3
		peers = [str(i) for i in range(10000)]
		peers2 = ["A" for _ in range(10000)]
		# initialize a random series of integers (generator object) that creates a unique series of integers
		random_gen  = random.Random(graph_seed)
		# initialize an instance of the MobilityTemplate
		mobility_template = MobilityTemplate(graph, N, T, graph_seed = graph_seed)

		# 1.1 Test the sample1 method
		# 1.1.1 sampled_peer
		self.assertEqual(mobility_template.sample1(random_gen, peers)[0], '2379' )
		# 1.1.2 random_gen state is not reset [that is we do not exclusively roll a single peer; therefore we sample again
		self.assertEqual(mobility_template.sample1(random_gen, peers)[0], '5442' )

		# re-initialize
		random_gen  = random.Random(graph_seed)
		mobility_template = MobilityTemplate(graph, N, T, graph_seed = graph_seed)
		# 1.1.3 remainder_peers, the third roll is "3699", thus the remainder_peers <list> is peers without "3699"
		self.assertEqual(mobility_template.sample1(random_gen, peers)[2], [el for el in peers if el != "2379"] )

		# re-initialize
		random_gen  = random.Random(graph_seed)
		mobility_template = MobilityTemplate(graph, N, T, graph_seed = graph_seed)

		# 1.2 Test the sampleN method
		# 1.2.1 Correct random_sample  :: random_gen, N, peers, must_sample_list = None, drop_duplicates = False
		self.assertEqual(mobility_template.sampleN(random_gen, N, peers)[0], ['2378', '5444', '3700'] )
		# 1.2.2 random_gen state is not reset [that is we do not exclusively roll the same peers]; therefore we sample again
		self.assertEqual(mobility_template.sampleN(random_gen, N, peers)[0], ['6041', '6259', '654'] )

		# re-initialize
		random_gen  = random.Random(graph_seed)
		mobility_template = MobilityTemplate(graph, N, T, graph_seed = graph_seed)

		# 1.2.3 kwarg 'must_sample_list' works correctly if the must_sample list <= N
		self.assertEqual(mobility_template.sampleN(random_gen, N, peers, must_sample_list = ['1', '2'])[0], ['1', '2', '2381'] )

		# re-initialize
		random_gen  = random.Random(graph_seed)
		mobility_template = MobilityTemplate(graph, N, T, graph_seed = graph_seed)

		# 1.2.4 kwarg 'must_sample_list' works correctly if the must_sample list > N
		self.assertEqual(mobility_template.sampleN(random_gen, N, peers, must_sample_list = ['1', '2', '3', '4', '5'])[0], ['1', '2', '3'] )

		# re-initialize
		random_gen  = random.Random(graph_seed)
		mobility_template = MobilityTemplate(graph, N, T, graph_seed = graph_seed)

		# 1.2.5 kwarg 'must_sample_list' is included in the random_sample even if the elements in must_sample_list are not in peers
		self.assertEqual(mobility_template.sampleN(random_gen, N, peers, must_sample_list = ['a', 'b', 'c', 'd'])[0], ['a', 'b', 'c'] )

		# 1.2.6 kwarg 'drop_duplicates' drops all duplicates before prompting the random_sample
		# the expected result with drop_duplicates == False is ['A', 'A', 'A']
		self.assertEqual(mobility_template.sampleN(random_gen, N, peers2, drop_duplicates = True)[0], ['A'] )










if __name__ == '__main__':
	# Run unit tests
	unittest.main()
