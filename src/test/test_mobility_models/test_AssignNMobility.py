# import testing module
import unittest

# add upper directory to sys-path to find graph.py
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[1]) # ./src
root_dir   = str(file_dir.parents[2]) # ./
sys.path.insert(0, src_dir)


from mobility_models.graph import DAG
from mobility_models.AssignNMobility import AssignNMobility

from collections import deque


class TestAssignNMobility(unittest.TestCase):
	""" Test whether the AssignNMobility produces the desired execution graph. We will consider the Example given in the Class's doc-string.

		Example:
		########
		- graph_dict= None
		- peers		= ["A", "B", "C"]

		- graph = DAG(graph_dict, peers, T=T)
		- N 	= 1
		- T		= 2
		- random_seed = 1234

		- mobility_model = AssignNMobility(graph, N, T)

		Then

		mobility_model.generate_graph()

		produces the following DAG (<mobility_model.graph>)

		A:0		B:0		C:0		^
		  \_____________/		|
          /             \		|
		A:1		B:1		C:1		| flow direction
		  \_____/				|
		  /     \				|
		A:2		B:2		C:2		|


		# 3. Test Graph Generation

		# 3.1 Correct vertices
		# 3.2 Correct edges
		# 3.3 Correct children
		# 3.4 Correct history

	"""
	###############################
	# 3. Graph Generation         #
	###############################

	def test_generate_graph(self):
		""" Test wether the graph generation works correctly. """
		N = 1
		T = 2
		graph_seed = 1234
		graph_dict = None
		peers = ["A", "B", "C"]
		graph = DAG(graph_dict, peers, T = T)
		# up until now graph has to be correct, else the following will not yield correct testing results
		mobility_model = AssignNMobility(graph, N, T, graph_seed = graph_seed)
		mobility_model.generate_graph()

		# 3.1 Correct vertices
		self.assertEqual(mobility_model.graph.vertices(), ['A:0', 'B:0', 'C:0', 'A:1', 'C:1', 'B:1', 'A:2', 'B:2', 'C:2'])
		# 3.2 Correct edges
		self.assertEqual(mobility_model.graph.edges(), [('A:1', 'C:0'), ('C:1', 'A:0'), ('A:2', 'B:1'), ('B:2', 'A:1')])
		# 3.3 Correct children
		self.assertEqual(mobility_model.graph.children('A:2'),deque(['B:1']))
		self.assertEqual(mobility_model.graph.children('C:2'),deque([]))
		# 3.4 Correct history
		self.assertEqual(mobility_model.graph.collect_history('A:2'), ['B:1', 'C:0'])


if __name__ == '__main__':
	# Run unit tests
	unittest.main()
