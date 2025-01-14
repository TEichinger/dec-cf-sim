from abc import ABCMeta

import numpy as np

class AggregatePayloadTemplate(metaclass = ABCMeta):
	""" Template Class for the <aggregate_payload> abstract method in the DecAlgoTemplate class (see 6. II.1.2 AGGREGATE PAYLOAD). The behavior is determined by the
	<algo_string>.


	Example:

		A payload to aggregate.

		Remarks:
			(1) Review the <CollectPayloadTemplate.fill_in_missing_columns> method on how the payload columns are filled.
			(2) The output aggregated_payload_list will be concatenated in the <DecAlgoTemplate.inner_for_loop> method.

		Input:
			payload:

					userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
				0	1		1		1.0		0.786889	NaN		2		0.786889		1
				1	1		2		5.0		0.786889	NaN		2		0.786889		1
				2	1		5		3.0		0.786889	NaN		2		0.786889		1
				3	1		6		2.5		0.786889	NaN		2		0.786889		1
				4	1		1		1.0		0.533333	NaN		2		0.786889		1
				5	1		2		1.0		0.533333	NaN		2		0.786889		1
				6	1		3		4.5		0.533333	NaN		2		0.786889		1
				7	1		4		2.0		0.533333	NaN		2		0.786889		1


		The result should look like:

		Output:
			aggregated_payload_list (e.g. for DecShCFv2):

				[
						userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
					0	1		1		1.0		0.786889	NaN		2		0.786889		1,
						userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
					1	1		2		5.0		0.786889	NaN		2		0.786889		1,
						userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
					2	1		3		4.5		0.533333	NaN		2		0.786889		1,
						userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
					3	1		4		2.0		0.533333	NaN		2		0.786889		1,
						userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
					3	1		5		3.0		0.786889	NaN		2		0.786889		1,
						userId	itemId	rating	sim			agg_sim	sender	sim_to_sender	timestamp
					3	1		6		2.5		0.786889	NaN		2		0.786889		1,

				]


	"""

	def __init__(self):
		pass


	def aggregate_payload(self, payload, peer):
		"""
			- peer: receiver of the aggregated payload
		"""

		# if the payload is empty, return the default empty snapshot row
		if payload.shape[0]==0:
			# we omit concatenation here and concatenate once in <fill_snapshots>/<fill_snapshots_inparallel> for speedup
			# if no rows (per itemId) have been made, make it an empty dataframe
			return self.default_snapshot_row()

		if self.algo_string in ("DecAggCFv2", "DecAggCFv5"):
			""" Aggregate the pandas dataframes of the format
			 into a single aggregated rating dataframe

			NOTE that the output is not a single line, since it is in coordinate format is contains as many rows as there are unq rated items
			NOTE that the "userId" column is constant per payload
			NOTE that the "sender" and "sim" columns are in a 1:1 relation, i.e. a particular sender always has the same "sim" value and vice versa
			NOTE that the "sender"s correspond to the sets W_j and Omega_j in the paper (per item j)
			NOTE that the "rating" column is either an aggregated rating if "agg_sim" is NaN, or a raw rating is "agg_sim" is not NaN
			Aggregating formulae:


			Input :
				- payload (pandas dataframe; output of <collect_payload>
				NOTE a payload is the raw information that should be aggregated by a single sender (child)
			Output:
				- aggregated_payload_list : list of aggregated rating pandas.DataFrame per itemId
				NOTE that the "sim" column is deliberately NaN, since the similarity in this column should refer to the similarity between the child_child to the parent,
				which changes in every data exchange. Thus after the exchange of the aggregated rating information of this function, the parent (receiver) will become the child (sender)
				and the child will become a child_child (unless the sim to the new receiver is too low)

			"""
			# select the "sim" value if the "agg_sim" value is NaN, else select the "agg_sim" value
			payload.loc[:,"agg_sim"] = payload.apply(lambda x: x["sim"] if x.isna()["agg_sim"] else x["agg_sim"], axis = 1)

			# 1. sim*rating -> rating: the "rating" column now holds per row products of "sim" and "rating"
			payload.loc[:,"rating"] = payload.apply(lambda x: x["agg_sim"]*x["rating"], axis = 1)

			# 2. sum "sim" and "rating" (=sim*rating) columns; pick the first item for the other columns
			# NOTE that for every itemId-group, the entries are identical per column (cf. <CollectPayloadTemplate.fill_in_missing_columns>)
			payload= payload.groupby("itemId", as_index = False).agg({"userId": "first",\
														"rating": "sum",\
														"sim": "first",\
														"agg_sim": "sum",\
														"sender": "first",\
														"sim_to_sender": "first",\
														"timestamp": "first"})
			# reorder columns to fit the default (effectively switch "userId" and "itemId")
			payload = payload[['userId', 'itemId', 'rating', 'sim', 'agg_sim', 'sender', 'sim_to_sender', 'timestamp']]

			# 3. sum_itemId(sim*rating) / sum_itemId(sim)
			payload.loc[:,"rating"] = payload.apply(lambda x: x["rating"]/x["agg_sim"], axis = 1)
			# fill in NaN value in "sim", the receiver should not know any similarity to the payload profiles
			payload.loc[:,"sim"] = np.nan


		elif self.algo_string == "DecCF":
			"""Aggregate a payload (pandas DataFrame). Since in the vanilla version of the algorithm, no actual aggregation takes place, this is an empty function
				that puts every payload into a list.
			Input:
				- payload : pandas.DataFrame (of information to aggregate; not in this case)
			Output:
				- aggregated_payload_list: list of pandas.DataFrame	(list of payload)
			"""
			pass


		elif self.algo_string in ("DecShCFv2", "DecShCFv3"):
			""" Aggregate the pandas dataframes of the format
				into a single aggregated rating dataframe

				NOTE that the output is not a single line, since it is in coordinate format is contains as many rows as there are unq rated items
				NOTE that the "userId" column is constant per payload
				NOTE that the "sender" and "sim" columns are in a 1:1 relation, i.e. a particular sender always has the same "sim" value and vice versa
				NOTE that the "sender"s correspond to the sets W_j and Omega_j in the paper (per item j)
				NOTE that the "rating" column is either an aggregated rating if "agg_sim" is NaN, or a raw rating is "agg_sim" is not NaN
				Aggregating formulae:
					Follows the formula as per the paper.

			Input :
				- payload (pandas dataframe; output of <collect_payload>
				NOTE a payload is the raw information that should be aggregated by a single sender (child)
			Output:
				- aggregated_payload_list : list of aggregated rating pandas.DataFrame per itemId
				NOTE that the "sim" column is deliberately NaN, since the similarity in this column should refer to the similarity between the child_child to the parent,
				which changes in every data exchange. Thus after the exchange of the aggregated rating information of this function, the parent (receiver) will become the child (sender)
				and the child will become a child_child (unless the sim to the new receiver is too low)
			"""
			# (x)
			# sort the payload by "itemId" and then by "sim" such that
			# the dataframe has block-wise subdataframes for the same item
			payload = payload.sort_values(by=["itemId", "sim"], ascending = False)
			# then, for every itemId pick the "first" encountered entry that represents the rating of the most similar child_child (to parent)
			payload = payload.drop_duplicates(subset = ["itemId"], keep = "first")


		elif self.algo_string in ("DecAggCFv6"):
			""" Aggregate the pandas dataframes of the format
				into a single aggregated rating dataframe without using aggregated similarities.


				NOTE that the output is not a single line, since it is in coordinate format is contains as many rows as there are unq rated items
				NOTE that the "userId" column is constant per payload
				NOTE that the "sender" and "sim" columns are in a 1:1 relation, i.e. a particular sender always has the same "sim" value and vice versa
				NOTE that the "sender"s correspond to the sets W_j and Omega_j in the paper (per item j)
				NOTE that the "rating" column is either an aggregated rating if "agg_sim" is NaN, or a raw rating is "agg_sim" is not NaN
				Aggregating formulae:
					Follows the formula as per the paper.

			Input :
				- payload (pandas dataframe; output of <collect_payload>
				NOTE a payload is the raw information that should be aggregated by a single sender (child)
			Output:
				- aggregated_payload_list : list of aggregated rating pandas.DataFrame per itemId
				NOTE that the "sim" column is deliberately NaN, since the similarity in this column should refer to the similarity between the child_child to the parent,
				which changes in every data exchange. Thus after the exchange of the aggregated rating information of this function, the parent (receiver) will become the child (sender)
				and the child will become a child_child (unless the sim to the new receiver is too low)
			"""

			# 1. sim*rating -> rating: the "rating" column now holds per row products of "sim" and "rating"
			payload.loc[:,"rating"] = payload.apply(lambda x: x["sim"]*x["rating"], axis = 1)

			# 2. sum "sim" and "rating" (=sim*rating) columns; pick the first item for the other columns
			# NOTE that for every itemId-group, the entries are identical per column (cf. <CollectPayloadTemplate.fill_in_missing_columns>)
			payload= payload.groupby("itemId", as_index = False).agg({"userId": "first",\
														"rating": "sum",\
														"sim": "sum",\
														"agg_sim": "first",\
														"sender": "first",\
														"sim_to_sender": "first",\
														"timestamp": "first"})
			# reorder columns to fit the default (effectively switch "userId" and "itemId")
			payload = payload[['userId', 'itemId', 'rating', 'sim', 'agg_sim', 'sender', 'sim_to_sender', 'timestamp']]

			# 3. sum_itemId(sim*rating) / sum_itemId(sim)
			payload.loc[:,"rating"] = payload.apply(lambda x: x["rating"]/x["sim"], axis = 1)
			# fill in NaN values in "sim", the receiver should not know any similarity to the payload profiles (see also CollectPayloadTemplate.<fill_in_missing_columns>)
			payload.loc[:,"sim"] = np.nan
			# fill in NaN values in "agg_sim" as no aggregated similarities are used (see also CollectPayloadTemplate.<fill_in_missing_columns>)
			payload.loc[:,"agg_sim"] = np.nan

		else:
			print("aggregate_payload method could not be properly identified. Checkout AggregatePayloadTemplate.py and the algorithm-class for proper linking via the algo_string.")


		aggregated_payload = payload

		# OPTIONAL: center mean of aggregated profile
		# RECALL that in DecCF, we have snapshots in which the rating originator is given in the "userId" column
		# in contrast, the "userId" for the remainder of algorithms (DecAggCFv2/5/6, DecShCFv2/3), userId is identical to the "sender"

		if self.mean_centering:
			if self.algo_string in ("DecCF"):
				# NOTE that for self.algo_string == "DecCF", all profiles already have zero mean ratings.
				# As no profile aggregation is involved, mean centering of the aggregated payload is superfluous as it already has mean zero
				# as a concatenation of profiles of mean zero.
				pass
			elif self.algo_string in ("DecAggCFv2", "DecAggCFv5", "DecAggCFv6", "DecShCFv2", "DecShCFv3"):
				# there is only one userId (= receiver)
				aggregated_payload = self.center_mean(aggregated_payload)

			# add mean rating of receiver (peer)
			aggregated_payload.loc[:,"rating"] = aggregated_payload.loc[:,"rating"] + self.mean_rating_dict[peer]




		return aggregated_payload
