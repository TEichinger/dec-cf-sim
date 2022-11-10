
# for production
from abc import ABCMeta, abstractmethod

class ParameterControlTemplate(metaclass=ABCMeta):
	"""
	This abstract class defines the interface of simulation paramters (including for instance) time horizon <T> and <percentile>
	toward the abstract <DecAlgoTemplate> class. The idea is that this class holds parameter values per peer. It further allows to update
	these individual parameter values according to a specific update scheme. Examples are for instance constant parameters which is equivalent
	to no updates to the initial parameters, but can also be extended to enable for instance Distributed Gradient Descent-based parameter updates.

	Workflow:
	    1. Initialize an experiment by intializing an instance of the DecAlgoTemplate (e.g. via DecAggCFv2.py)
	    2. Run the simulation by calling the DecAlgoTemplate.fill_snapshots() method of the DecAlgoTemplate class instance.
	        2.1 The algorithm will then run DecAlgoTemplate.inner_for_loop() for every epoch and for every user.
	        2.2 Between epochs, the parameters (graph, and min_sim_to_sender) are updated by calling the ParameterControlTemplate.update() method.
	            More precisely, the parameters are updated in the ParameterControlTemplate class instance and then copied into the DecAlgoTemplate class instance.



	REMARK(S):
		1.  Depending on whether or not parameter tuning depends on system performance (see Workflow 2.2), you may need to add a validation
	        dataset as well as the current future_snapshots_dict (see: ADD EXAMPLE).


	Main methods:
	    <self.update_parameters>: update the simulation parameters for every user;
	                                this mainly updates <self.graph> and <self.min_sim_to_sender> to minimize future connections and users' similarity thresholds.

	Parameters:

	- graph					: <list> of all peers' identifiers specified in the train_df; [see column "userId"]; the execution graph in the abstract DecAlgoTemplate class
	- dynamic_percentile	: <float> percentile of a peer's similarity histogram to use as threshold for sending data if above, if None: use the median
	- min_sim_to_sender_dict: <dict> of <float> similarity thresholds per user (format: {userId: sim_threshold_of_userId}) from the abstract DecAlgoTempalte class.

	Optional parameters:
	- val_df                : validation_df, a dataframe that holds ratings.
	                            NOTE Only required if parameter updates depend on the performance achieved on this dataframe.

	"""

	def __init__(self, algorithm, val_df=None):
		self.algorithm = algorithm
		# [OPTIONAL] control parameters on the basis of performance on the validation dataframe <val_df>
		self.val_df                             = val_df


	def update_parameters(self, t, new_future_snapshots_dict_entries):
		""" Calculate novel simulation parameters <graph> and <min_sim_to_sender_dict> for an instance of the <DecAlgoTemplate> class."""
