from abc import ABCMeta

import numpy as np


class CollectPayloadTemplate(metaclass = ABCMeta):
	""" Template Class for the <collect_payload> abstract method in the DecAlgoTemplate class (see 6. II.1.1 COLLECT PAYLOAD). The behavior is determined by the
	algo_string. """

	def __init__(self):
		pass

	def collect_payload(self, parent, child, t, timestamp_delta):
		""" Bundles behavior of the <collect_payload> method of the Algorithms with algo_strings ("DecAggCFv2", "DecAggCFv5", "DecShCFv2", "DecShCFv3", "DecCF")
		For faster Search:

			(I.1) <collect_payload> for "DecAggCFv2", "DecAggCFv5", "DecAggCFv6" "DecShCFv2", "DecShCFv3"
			(I.3) 					for "DecCF"

			(II.1) <fill_in_missing_columns> for "DecAggCFv2", "DecAggCFv5", "DecCF", "DecShCFv2", "DecShCFv3"

		"""
		# (I.1)
		if self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecShCFv2", "DecShCFv3", "DecCF"):

			""" Collect the topN profiles in the child's (sender's) snapshot that are most similar to the parent (receiver).
			POTENTIAL SPEEDUP: Build a dictionary of pairwise similarities such that recalculations of similarity for
							   future timesteps are obsolete

			Nov 29, 2019: I have checked every substep in this function except the correct imputation from self.__future_snapshots_df
			since filling this requires the aggregation step!

			Input:
				- parent					: <str> ID of the receiver (also refered to as parent)
				- child						: <str> ID of the sender of the data (also refered to as child)
				- t							: <int> epoch of data exchange
												NOTE	that t is the epoch of data exchange; therefore only profiles collected until epoch (t-1) are considered for
														exchange. Within the epoch t no order between contacts is foreseen.
				- timestamp_delta			: <int> maximal difference in epochs (as a timestamp unit) to consider previously received profile information
											  for aggregation. note that entries that date back longer than timestamp_delta epochs are not actively deleted
											  from the future snapshots!

			Output:
				- payload					: pandas.DataFrame that contains all the information to be included into the aggregation procedure
											  NOTE that the pandas DataFrame has already set the
											  "userId" to the parent (= receiver)
											  "sender" to the child
											  "sim"    is the similarity of the parent to the child_child (not included as information)
											  "agg_sim" stays untouched
											  "sim_to_sender" is just additional information that will currently not be included into either filtering or aggregation

			Example:

			payload

				userId itemId rating        sim    agg_sim sender sim_to_sender	timestamp
			0        7      1      2   0.111499        NaN      5      0.111499			3
			1        7      2      3   0.111499        NaN      5      0.111499			3
			2        7     10      5   0.111499        NaN      5      0.111499			3
			3        7     15      1   0.111499        NaN      5      0.111499			3
			4        7      1      1  0.0537215  0.0494166      5     0.0537215			4
			5        7      3      5  0.0537215  0.0494166      5     0.0537215			4
			6        7      4      4  0.0537215  0.0494166      5     0.0537215			4

			NOTE that sender (child) and userId (parent) are constant over the payload, this reflects the fact that the sender mixes information from child_children
				 into a single aggregated profile represented by the sender only thus "hiding" the previously met child_children (and of course also his raw rating information).
			"""

			###########################################################################################################
			# 0. INITIALIZE VARIABLES                                                                                 #
			###########################################################################################################
			# fetch the child_df, the child_df is the basis for the payload [userId == child]
			child_df  = self.fetch_sub_train_df(child)
			# select the child's future snapshot df
			child_future_df = self.future_snapshots_dict[child]
			# if <timestamp_delta> is not None, filter for the entries that at maximum date back <timestamp_delta> epochs
			if timestamp_delta is not None: # TODO: slice this with .loc and also make it a copy?
				child_future_df = child_future_df[child_future_df["timestamp"]>=t-timestamp_delta]

			# Find child_children ...
			# NOTE	that depending on the execution graph, child may also by in his own history, if for instance ("child:t", "child:t-1") is an edge.
			#		See ./src/mobility_models/graph.py for details
			# Profile Aggregation.
			# We assume that profiles are anonymous for they represent multiple profiles. The "userId" column holds the data tenant's ID.
			# We can only distinguish distinct contacts (by a combination of "sender" and "timestamp", assuming that every user only uses one peer in the network).
			if self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecShCFv2", "DecShCFv3"):
				child_children_without_child	= [el.split(":")[0] for el in self.graph.collect_history(child+":"+str(self.t-1), timestamp_delta = timestamp_delta)]
				# ... and the corresponding timestamps at which the child and child_child met
				# (Y)
				# NOTE that the +1 is necessary since the then children are the now child_children
				# e.g.
				# THEN:
				#		("1:4", "2:3") ["1" (then parent) finds "2" (then child) at epoch 4 and claims profiles collected by "2" until epoch 3]
				# NOW:
				#		("3:100", "1:99") ["3" (now parent) finds "1" (now child) at epoch 100 and claims profiles collected by "1" util epoch 99]
				#							this includes the profile received at epoch 4 (see THEN). Since the collect_history returns the children including "2:3"
				#							we need increment the epoch 3 by one.
				# Recall that 	self.graph.collect_history(child+":"+str(t-1)) returns a list of "child:epoch" strings.
				child_children_without_child_timestamps	= [str(int(el.split(":")[1])+1) for el in self.graph.collect_history(child+":"+str(t-1), timestamp_delta = timestamp_delta)]
				# add child as child_child (*) always at the first (0-th position in the list)
				child_children = [child] + child_children_without_child
				# and a zero for the child
				child_children_timestamps = [0] + child_children_without_child_timestamps

				# if contacts share rating data non-anonymously, we can identify contacts with the same peer between epochs
				# since multiple contacts with the same peer lead to duplicates in child_children_without_child, we
				# (1) drop duplicates here
				# (2) select future snapshot dfs by "sender" instead of combinations of "sender" and "timestamp" in (X)
				#
				if not self.anonymous_contacts:
					# (1) drop duplicate children (child is always the first element)
					child_children					= list(dict.fromkeys(child_children)) # this version is slower than list(set(child_children)) yet is reproducible
					child_children_without_child	= child_children[1:]              # slice at 1 since we exactly impute one child (at index 0) (cf. (*) and (**))


			# No Profile Aggregation.
			# We assume that profiles are not anonymous, which is why the "userId" column holds the data originator's ID, and we drop duplicates.
			elif self.algo_string in ("DecCF"):
				child_children_without_child	= child_future_df.loc[:,"userId"].drop_duplicates().tolist()
				# filter out the parent's profile data, which could have been collected by child up until (t-1) via mutually similar other peers
				# consequently future_dfs do not contain any ratings by child
				child_children_without_child	= [child_child for child_child in child_children_without_child if child_child != parent]
				child_children = [child] + child_children_without_child


			###########################################################################################################
			# 1. CALCULATE SIMILARITY BETWEEN CHILD AND PARENT                                                        #
			###########################################################################################################

			# (********)
			# NOTE	that the child assesses the similarity sim(child,parent), therefore the order is (child, parent, t_str) instead of (parent, child, t_str)
			if self.hide_p is None:
				sim_parent_to_child_list = [self.fetch_similarity(child, parent)]
			else:
				pass
				#t_str= str(t)
				#sim_parent_to_child_list = [self.fetch_similarity(child, parent, t_str)]

			sim_parent_to_child = sim_parent_to_child_list[0]

			###########################################################################################################
			# 2.1 IF SIMILARITY BETWEEN CHILD(SENDER) AND PARENT(RECEIVER) IS HIGH ENOUGH                             #
			###########################################################################################################
			# PREVIOUSLY: [CHECKOUT] if (sim_parent_to_child > self.min_sim_to_sender_dict[parent]) & (sim_parent_to_child > self.min_sim_to_sender): [b1]
			# NOW:                   if (sim_parent_to_child > self.min_sim_to_sender_dict[child]) & (sim_parent_to_child > self.min_sim_to_sender):  [a1]
			# [parent] performs very much more efficiently!? WHY !?
			#                                                    parent: sender threshold; child: receiver threshold
			# CHANGED ON 22 March 2022                            ||||||
			#                                                     vvvvvv
			if (sim_parent_to_child > self.min_sim_to_sender_dict[parent]) & (sim_parent_to_child > self.min_sim_to_sender):

				#######################################################################################################
				# 2.1.1 CALCULATE CANDIDATE SIMILARITIES                                                              #
				#######################################################################################################
				if self.hide_p is None:
					if self.algo_string in ("DecAggCFv2", "DecAggCFv6", "DecShCFv2", "DecCF"):											# here child_children were unq_child_children
						payload_candidate_sims_without_child = [self.fetch_similarity(parent, child_child) for child_child in child_children_without_child]
					elif self.algo_string in ("DecAggCFv5", "DecShCFv3"):
						payload_candidate_sims_without_child = [self.fetch_similarity(child,  child_child) for child_child in child_children_without_child]
				else:
					# TODO: make this work for all experiments!
					########
					#if self.algo_string in ("DecAggCFv5"):
					#	# RECALL that the signature for the DecAlgoTemplate.fetch_similarity method is (sender, recipient, timestamp)
					#	#		Since the now sender (now child) is the then recipient, and the then sender (now child_child) is the now sender, we have
					#	#		In other words, we fetch the similarity assessed by child, on the basis of profile data sent by child_child at timestamp
					#	payload_candidate_sims_without_child = [self.fetch_similarity(child, child_child, timestamp) for child_child, timestamp in zip(child_children_without_child, child_children_without_child_timestamps)]
					#elif self.algo_string in ("DecShCFv2"):
					#	II = None
					#elif self.algo_string in ("DecAggCFv2"):
					#	unq_child_children_without_child
					#	# we need to find the encounterkey, which represents the encounter between
					#	# the child_child (sender of the subsampled profile) and the child (receiver of the subsampled profile) for some time t
					#	# --> "child_child-child:t"
					#	# the references point at the rows known to the child about child_child at time t
					#	#print(self.initial_snapshot_refs())
					pass
				payload_candidate_sims = sim_parent_to_child_list + payload_candidate_sims_without_child


				#######################################################################################################
				# 2.1.2 COLLECT CANDIDATE DFS                                                                         #
				#######################################################################################################
				# if there are no child_children, the future part (*_without_child)is empty. Then only the child's raw profile information will be sent.
				if child_children_without_child == []:
					payload_candidate_dfs_without_child = []
				# else collect the future part from the future_snapshots_dict[child] entries, in particular those entries that have been sent by
				# child_child ["sender" == child_child] at a specific timestamp in order to distinguish between distinct encounters of the same child_child
				else: #                                                                                -1 is required since the timestamps are incremented by one (see (Y))
					# if profile aggregation is performed, the aggregated profile is represented by 'sender'
					if self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecShCFv2", "DecShCFv3"):
						# (X)
						if self.anonymous_contacts:
							# (1) select future snapshots dfs by distinct combinations of "sender" and "timestamp"
							#	  (multiple contacts between epochs cannot be distinguished)
							payload_candidate_dfs_without_child = [child_future_df.loc[(child_future_df["sender"] == child_child) & (child_future_df["timestamp"] == int(timestamp)-1)].copy() \
										for child_child, timestamp in zip(child_children_without_child, child_children_without_child_timestamps)]
						else:
							# (2) select future snapshot dfs by distinc "sender"
							#	  (multiple contacts between epochs can be distinguished)
							payload_candidate_dfs_without_child = [child_future_df.loc[(child_future_df["sender"] == child_child)].copy() \
													for child_child in child_children_without_child]
					# elif no profile aggregation is to be performed, we have to retrieve the individual profiles via "userId", which represents the data originator's ID
					elif self.algo_string in ("DecCF"):
						# here we assume again that profiles do not change over time, else this has to be adjusted
						payload_candidate_dfs_without_child = [self.fetch_sub_train_df(child_child) for child_child in child_children_without_child]


				#######################################################################################################
				# 2.1.3 FILL IN MISSING VALUES TO THE CANDIDATE_DFS                                                   #
				#######################################################################################################
				# fill in missing values to the payload_candidate_dfs to make them conform with <default_snapshot_columns/row> in the parent class
				# firstly for the child_df (initial_df)         [raw]

				# fill in missing values to the payload_candidate_dfs to make them conform with <default_snapshot_columns/row> in the parent class
				####################################################################################
				if self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecShCFv2", "DecShCFv3"):
					child_df 							= self.fill_in_missing_columns(child_df, parent, sim_parent_to_child, child, t, omit_agg_sim = False)
					# secondly for the child_child_dfs (future_dfs) [aggregated]
					payload_candidate_dfs_without_child	= [self.fill_in_missing_columns(candidate_df, parent, candidate_sim, child, t, omit_agg_sim = True) \
																for candidate_df, candidate_sim \
																in zip(payload_candidate_dfs_without_child, payload_candidate_sims_without_child)]
				elif self.algo_string in ("DecCF"):
					child_df							= self.fill_in_missing_columns(child_df, child, sim_parent_to_child, child, t, omit_agg_sim = False)
					# secondly for the child_child_dfs (future_dfs) [non-aggregated]
					payload_candidate_dfs_without_child	= [self.fill_in_missing_columns(candidate_df, child_child, candidate_sim, child, t, omit_agg_sim = False) \
																for candidate_df, candidate_sim, child_child \
																in zip(payload_candidate_dfs_without_child, payload_candidate_sims_without_child, child_children_without_child)]

				# finally, put both together
				payload_candidate_dfs = [child_df] + payload_candidate_dfs_without_child

				#######################################################################################################
				# 2.1.4 FILTER AND SORT CANDIDATE DFS BY CANDIDATE SIMILARITY                                         #
				#######################################################################################################      #(X) here unq_child_children
				candidate_sims_and_candidate_dfs_and_child_children = list(zip(payload_candidate_sims, payload_candidate_dfs, child_children))
				# sort by descending candidate_similarity and larger than self.min_sim_to_child_child (default = 0.0)
				positive_sorted_sims_and_candidate_dfs_and_child_children = [(candidate_sim, candidate_df, child_child) for candidate_sim, candidate_df, child_child in \
																sorted(candidate_sims_and_candidate_dfs_and_child_children, \
																	key=lambda x: x[0], reverse = True)	if candidate_sim > self.min_sim_to_child_child]
				if self.topN is None:
					topN_positive_sorted_sims_and_candidate_dfs_and_child_children = positive_sorted_sims_and_candidate_dfs_and_child_children
				else:
					topN_positive_sorted_sims_and_candidate_dfs_and_child_children = positive_sorted_sims_and_candidate_dfs_and_child_children[:self.topN]


				#######################################################################################################
				# 2.1.5 CONCATENATE CANDIDATE DFS INTO THE PAYLOAD                                                    #
				#######################################################################################################

				# persist the topN positive sorted sims and candidate dfs in self.future_snapshots_dict
				topN_positive_candidate_dfs = [df  			for sim,df,child_child in topN_positive_sorted_sims_and_candidate_dfs_and_child_children]
				#topN_positive_sims			= [sim 			for sim,df,child_child in topN_positive_sorted_sims_and_candidate_dfs_and_child_children]
				#topN_child_children			= [child_child	for sim,df,child_child in topN_positive_sorted_sims_and_candidate_dfs_and_child_children]

				# OPTIONAL: center means of candidate profiles
				if self.mean_centering:
					topN_positive_candidate_dfs = [self.center_mean(profile) for profile in topN_positive_candidate_dfs]

				# concatenate the topN_positive_candidate_dfs
				payload = self.fast_concat(topN_positive_candidate_dfs)

			###########################################################################################################
			# 2.2 ELSE FORM AN EMPTY PAYLOAD                                                                          #
			###########################################################################################################
			else:
				# else return and empty DataFrame
				payload = self.default_snapshot_row()

			return payload



	def fill_in_missing_columns(self, candidate_df, ref_userId, candidate_sim, child, t, omit_agg_sim = False):
		# (II.1)
		if self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecCF", "DecShCFv2", "DecShCFv3"):
			""" Fill in the missing columns of a candidate_initial_df of future_snapshots_df such that the columns match those in <default_snapshot_columns>.

			Input:
				- candidate_df			: sub pandas.DataFrame from self.__train_df; or self.future_snapshots_dict
				- ref_userId			: <str> filling for the "userId" column
				- candidate_sim			: <float> similarity to be used in AggregatePayloadTemplate.aggregate_payload, filling for the "sim" column
				- child					: <str> child ID, filling of the "sender" column
				- t						: <int> epoch of data exchange (reception of the profile data in the snapshot_df)
				- omit_agg_sim			: <bool> if True: keep the agg_sim value, else set to default np.nan
			Output:
				- result_candidate_initial_df : pandas.DataFrame with filled in columns
											"userId"		--> ref_userId	:
																if DecAggCFv2, DecAggCFv5, DecShCFv2, DecShCFv3 : ref_userId = parent
																if DecCF 										: ref_userId = userId
																					(that is the data originator, in this case, we do not have anonymity,
																					every user is identifiable, which is necessary for dropping duplicates
																					in <collect_garbadge>
											"sim"			--> candidate_sim, where
																if DecAggCFv2, DecAggCFv6, DecShCFv2: candidate_sim = sim_parent_to_child_child
																if DecAggCFv5, DecShCFv3            : candidate_sim = sim_child_to_child_child
																if DecCF				            : candidate_sim = sim_parent_to_child_child
											"agg_sim"		--> np.nan; if omit_agg_sim == True: keep the present agg_sim value
											"sender"		--> child
											"sim_to_sender"	--> candidate_sim, where
																if DecAggCFv2, DecAggCFv5: sim(child, parent)
																if DecCF				 : sim(child, userId), where userId is the data originator

			TODO: CAVEAT, changing the order of <default_snapshot_columns> will make this wrong, make it failure-proof in the future!
			NOTE: (***) The meaning of "userId" in this implementation is the originator of the (profile) data, since we can infer! In contrast to "userId"
			in the aggregating implementation, where "userId" reflects the data tenant, since in all generality, since that is not a raw profile by any peer.
			"""
			t_str = str(t)
			# initialize default_columns
			default_columns = self.default_snapshot_columns()

			result_candidate_df = candidate_df

			for col_name in default_columns:
				# fill in all column entries except those for the columns "itemId" and "rating"
				if col_name not in ["itemId", "rating"]:
					if col_name == "userId":
						result_candidate_df["userId"] = ref_userId
					if col_name == "sim":
						result_candidate_df["sim"] = candidate_sim
					elif col_name == "agg_sim":
						if omit_agg_sim:
							continue
						else:
							result_candidate_df["agg_sim"] = np.nan
					elif col_name == "sender":
						result_candidate_df["sender"] = child
					elif col_name == "sim_to_sender":
						if self.hide_p is None:
							result_candidate_df["sim_to_sender"] = self.fetch_similarity(child, ref_userId)
						else:
							# child assesses the "sim_to_sender" for parent
							pass
							#result_candidate_df["sim_to_sender"] = 	self.fetch_similarity(child, ref_userId, t_str)
					elif col_name == "timestamp":
						result_candidate_df["timestamp"] = t

			return result_candidate_df
