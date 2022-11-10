# import testing module
import unittest

# add upper directory to sys-path to find graph.py
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[1]) # ./src
root_dir   = str(file_dir.parents[2]) # ./
sys.path.insert(0, src_dir)

# import the DAG class to test
from mobility_models.graph import DAG

# import statements from graph.py
from collections import deque


class TestDAG(unittest.TestCase):
	""" Test the following functionalities of the DAG (directed acyclic graph) class in the graph.py file
		0. Test return functions
			0.a Return graph_dict
			0.b Return peers
			0.c Return T
			0.d Return vertices
			0.e Return edges
			0.f Return children
		1. Test update functions
			1.a Update the depth of the DAG (= self.T)
			1.b Update the list of vertices to include the peers at time 0
		2. Test graph building functions
			2.a Add vertex
			2.b Add edge
			2.c Collect History (the sub-DAG for a given root-vertex)

	"""
	#############################
	# 0. Test return functions  #
	#############################

	def test_return_functions(self):
		""" Test whether the return functions return the correct elements of the DAG. """
		graph_dict = { "A:0":             deque([]),
				"B:0":             deque([]),
				"C:0":             deque([]),
				"A:1": deque(["A:0", "C:0"]),
				"B:1":        deque(["B:0"]),
				"C:1": deque(["A:0", "C:0"])
				}
		peers = ["D", "E"]
		test_dag = DAG(graph_dict = graph_dict, peers = peers)

		# 0.a Return graph_dict
		self.assertEqual(test_dag.graph_dict(), graph_dict)
		# 0.b Return peers
		self.assertEqual(test_dag.peers(), peers)
		# 0.c Return T
		self.assertEqual(test_dag.T(), None)
		# 0.d Return vertices
		self.assertEqual(test_dag.vertices(), ['A:0', 'B:0', 'C:0', 'A:1', 'B:1', 'C:1'])
		# 0.e Return edges
		self.assertEqual(test_dag.edges(), [('A:1', 'A:0'), ('A:1', 'C:0'), ('B:1', 'B:0'), ('C:1', 'A:0'), ('C:1', 'C:0')])
		# 0.f Return children
		self.assertEqual(test_dag.children('A:0'), deque([]))
		self.assertEqual(test_dag.children('A:1'), deque(['A:0', 'C:0']))
	
	############################
	# 1. Test update functions #
	############################
	
	def test_update_functions(self):
		""" Test whether the update function return the correct outputs."""
		graph_dict = { "A:0":             deque([]),
				"B:0":             deque([]),
				"C:0":             deque([]),
				"A:1": deque(["A:0", "C:0"]),
				"B:1":        deque(["B:0"]),
				"C:1": deque(["A:0", "C:0"])
				}
		peers = ["D", "E"]
		test_dag = DAG(graph_dict = graph_dict, peers = peers)

		# 1.a Update the depth of the DAG self.T
		test_dag.update_T()
		self.assertEqual(test_dag.T(), 1)
		# 1.b Update the list of vertices to include the peers at time 0
		test_dag.initialize_peers() # add the vertices "D:0" and "E:0"
		self.assertEqual(test_dag.peers(), peers)

	######################################
	# 2. Test graph building functions   #
	######################################

	def test_graph_building_functions(self):
		""" Test the graph building functions. """
		graph_dict = { "A:0": deque([])}
		test_dag = DAG(graph_dict = graph_dict)
		source_vertex = "A:1"
		target_vertex = "A:0"		

		# 2.a Add vertex
		test_dag.add_vertex(source_vertex)
		self.assertEqual(test_dag.vertices(), [target_vertex, source_vertex])

		# 2.b Add edge
		test_dag.add_edge(source_vertex, target_vertex)
		self.assertEqual(test_dag.edges(), [('A:1', 'A:0')])

		# 2.c Collect History
		self.assertEqual(test_dag.collect_history(target_vertex), [])
		self.assertEqual(test_dag.collect_history(source_vertex), ['A:0'])
		self.assertEqual(test_dag.collect_history(source_vertex, timestamp_delta = 0), [])
		self.assertEqual(test_dag.collect_history(source_vertex, timestamp_delta = 1), ['A:0'])
		
		
if __name__ == '__main__':
	# Run unit tests
	unittest.main()
