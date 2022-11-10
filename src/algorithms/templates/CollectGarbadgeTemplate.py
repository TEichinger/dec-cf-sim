from abc import ABCMeta



class CollectGarbadgeTemplate(metaclass = ABCMeta):
	""" Template Class for the <collect_gardbadge> abstract method in the DecAlgoTemplate class (see 6. II.1.3 COLLECT GARBADGE). The behavior is determined by the
	algo_string. """

	def __init__(self):
		pass

	def collect_garbadge(self, snapshots_dict_entry):
		if self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecShCFv2", "DecShCFv3"):
				""" Remove unnecessary information from the new_future_snapshots_dict_entry. """
				return snapshots_dict_entry

		elif self.algo_string == "DecCF":
			# in this (vanilla) version of decentralized CF, we do not have anonymity
			# since every user can be identified, it is not necessary to keep duplicates
			# CAVEAT	this assumes that user profiles do not change themselves over time!
			# 			Else this will lead to incorrect results.
			#snapshots_dict_entry = snapshots_dict_entry.drop_duplicates(subset = ["userId", "itemId"], inplace = False)
			# NOTE here that userId is not (!) the data originator, but currently the parent (the receiver)
			# for instance we can drop duplicates on the basis of "sim" which is the sim of the parent to the child_child (where child_child is the data originator)
			snapshots_dict_entry = snapshots_dict_entry.drop_duplicates(subset = ["itemId", "rating", "sim"], inplace = False)
			return snapshots_dict_entry
		else:
			print("collect_garbadge method could not be properly identified. Checkout CollectGarbadgeTemplate.py and the algorithm-class for proper linking via the algo_string.")
