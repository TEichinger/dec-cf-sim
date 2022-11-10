
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)

from parameter_control.ParameterControlTemplate import ParameterControlTemplate

class StaticParameters(ParameterControlTemplate):
	""" Class that defines a static parameters for abstract DecAlgoTemplate class instances. That is, DecAlgoTemplate.graph and
	DecAlgoTemplate.min_sim_to_sender_dict do not change over time.

	NOTE that no validation set (val_df) is required, as parameters are static.

	"""
	#################################
	# 1. INITIALIZATION FUNCTIONS   #
	#################################
	def __init__(self, algorithm):
		# call the template's initialization
		super().__init__(algorithm, val_df = None)
		# parameter control string for future reference
		self.parameter_control_string	= "static"

	########################################################
	# 2. UPDATE SIMULATION PARAMETERS [MAIN FUNCTIONALITY] #
	########################################################

	def update_parameters(self, t, new_future_snapshots_dict_entries):
		""" Do not change neither self.graph not self.dynamic_percentile_dict
		"""
		# (I.1) UPDATE time horizon (T)
		# get current time horizons
		new_time_horizon_dict = self.algorithm.get_time_horizon_dict()
		# set new time horizons
		self.algorithm.set_time_horizon_dict(new_time_horizon_dict)

		# (I.2) UPDATE dynamic_percentile (1-dissemination parameter theta)
		# get current dynamic percentiles
		new_dynamic_percentile_dict = self.algorithm.get_dynamic_percentile_dict()
		# set new dynamic percentiles
		self.algorithm.set_dynamic_percentile_dict(new_dynamic_percentile_dict)

		return



if __name__ == "__main__":
	StaticParameters()
