
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)

import pandas as pd
import numpy as np
import math
from collections import defaultdict
from multiprocessing import Pool
from itertools import product, chain
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import coo_matrix, csr_matrix

from mobility_models.graph import DAG
from utilities.util import load_pickled_instance, drop_nan, safe_mean, load_ratings_df, make_train_test_split, load_snapshot_df, get_current_timestamp




def log_result_table(result_df, experiment_name):
	""" 
		Log the evaluation results in the result_df to a distinct log file in the form of a *.csv file to './data/logs/<log.csv>'
	"""
	# set output directory
	log_dir = os.path.join(root_dir,'data/evaluation_logs')

	# create name of the log.csv file (e.g. 2021_11_15_15_36_15_<experiment_name>.csv)
	log_name = ""
	current_time_string = get_current_timestamp()
	if current_time_string != "":
		log_name += current_time_string

	if experiment_name != "":
		log_name += '_' + experiment_name
	
	log_name += ".csv"

	# if the output directory does not exist, create the directory
	if not os.path.isdir(log_dir):
		os.makedirs(log_dir)

	# 
	log_path = os.path.join(log_dir, log_name)

	# 
	result_df.to_csv(log_path, index=False)

	return 



def count_data_exchanges(snapshot_output_path):
	""" For a given snapshot_output_path, iterate over all snapshots (.csv) files in the directory and count the number of data exchanges
		where a data exchange is identifiable by a unique combination of "sender" and "timestamp".

	Input:
		- snapshot_output_path	:
	Output:
		-


	"""
	number_data_exchanges_dict = dict()

	for root, dirs, files in os.walk(snapshot_output_path):
		for name in files:
			snapshot_path = os.path.join(root, name)
			if os.path.splitext(snapshot_path)[1] == ".csv":
				peer = os.path.splitext(snapshot_path)[0]
				peer_df = pd.read_csv(snapshot_path, index_col = 0)
				peer_df_unique_data_exchanges = peer_df.drop_duplicates(subset = ["sender", "timestamp"])
				peers_number_of_data_exchanges = peer_df_unique_data_exchanges.shape[0]
				number_data_exchanges_dict[peer] = peers_number_of_data_exchanges

	sum_of_data_exchanges = sum(list(number_data_exchanges_dict.values()))
	median = np.median(list(number_data_exchanges_dict.values()))
	av_data_exchanges_per_user = sum_of_data_exchanges/len(number_data_exchanges_dict)

	print("Sum of data exchanges: {} in {} with an average {} and median {} data exchanges per user.".format(\
											sum_of_data_exchanges, snapshot_output_path, av_data_exchanges_per_user, median))
	return



def count_entries(snapshot_output_path, min_time_stamp = None):
	""" For a given snapshot_output_path, iterate over all snapshots (.csv) files in the directory and count the number of rows
	(without counting the headline. Then take the average.

	"""
	number_of_entries_dict = dict()

	for root, dirs, files in os.walk(snapshot_output_path):
		for name in files:
			snapshot_path = os.path.join(root, name)
			if os.path.splitext(snapshot_path)[1] == ".csv":
				peer = os.path.splitext(snapshot_path)[0]
				peer_df = pd.read_csv(snapshot_path, index_col = 0)

				if min_time_stamp is not None:
					peer_df = peer_df[peer_df["timestamp"]>= min_time_stamp]

				peers_number_of_entries = peer_df.shape[0]
				number_of_entries_dict[peer] = peers_number_of_entries

	sum_of_entries = sum(list(number_of_entries_dict.values()))

	print("sum of entries {} in {} with min_time_stamp = {}".format(sum_of_entries, snapshot_output_path, min_time_stamp))
	return




def get_aggregated_payload_to_oneself(snapshot_path):
	""" For a given snapshots csv, calculate what would be sent (as an aggregated profile of the snapshot holder
		to himself).

		Input:
			- snapshot_path: <str> to the snapshot (.csv) file

		Output:
			- pandas.Series with itemIds (to be theoretically sent)
		"""
	# read the snapshot .csv file
	snapshot_df = pd.read_csv(snapshots_path, dtype = {'userId': str, 'itemId': str} )
	# pick the topN most similar users in the snapshot
	topN_snapshot_profiles
	# pick the unique itemIds of the topN_snapshot_rpfoiles


def surprise_predictions_to_predictions_dict(surprise_predictions):
	""" Take the predictions (list) as the output of a surprise algorithm, and transform it into a predictions dict.
		as is the output of <make_predictions_dict>. """

	predictions_dict= dict()

	userIds     		= []
	true_ratings 		= []
	estimated_ratings	= []

	for pred in surprise_predictions:
		userIds.append(pred.uid)
		true_ratings.append(pred.r_ui)
		estimated_ratings.append(pred.est)

	predictions_df = pd.DataFrame(list(zip(userIds, true_ratings, estimated_ratings)), columns = ["userId", "true_rating", "estimated_rating"])

	unq_userIds = predictions_df.loc[:,"userId"].drop_duplicates()

	for userId in unq_userIds:
		subframe = predictions_df[predictions_df["userId"] == userId]
		sub_rating_df = subframe.loc[:,["true_rating", "estimated_rating"]].to_numpy()
		predictions_dict[userId] = sub_rating_df

	return predictions_dict

def drop_entries(snapshot_df, drop_percentile = None, by = None):
	"""
	If <drop_percentile> is of type <float>:
		Drop the entries (rows of a coordinate ratings matrix) for which the <by> column holds
		values lower than the drop_percentile.

	Elif <drop_percentile> is of type <int>:
		Drop the entries (rows of a coordinate ratings matrix) that are not within the top <drop_percentile>
		values in the <by> column.

	"""
	if (drop_percentile is not None) & (by is not None) & (drop_percentile > 0):
		if type(drop_percentile) is float:
			# calculate the drop threshold (percentile)
			drop_threshold = np.percentile(snapshot_df.loc[:,by], drop_percentile*100)
			# cut off all entrie that are in the percentile (e.g. only keep those entries that represent the top 10%
			# "sim_to_sender"
			snapshot_df = snapshot_df[snapshot_df[by] > drop_threshold]
		elif type(drop_percentile) is int:
			num_by = len(snapshot_df.loc[:,by].drop_duplicates())
			if num_by >0:
				drop_threshold = snapshot_df.loc[:,by].drop_duplicates().sort_values(ignore_index=True, ascending = False)[min(drop_percentile-1, num_by-1)]
				# cut off all entrie that are in the percentile (e.g. only keep those entries that represent the top 10%
				# "sim_to_sender"
				snapshot_df = snapshot_df[snapshot_df[by] >= drop_threshold]


	return snapshot_df


def get_overall_coverage(predictions_dict):
	""" For a predictions_dict WITH NANS!, calculate the overall coverage as the fraction of all user-item pairs
	for which a prediction can be made (i.e. the estimated rating is not nan).
	Input:
		- predictions_dict: <dict> of the forma: {userId: 2D np.array([[true_rating, estimated_rating]])
	Output:
		- overall_coverage: <float> between 0 and 1 that represents the percentage of user-item pairs of the test set
							that allow rating prediction
	"""
	user_item_pair_counter		= 0
	prediction_possible_counter	= 0

	for peer in predictions_dict.keys():
		# predictions_df is a 2D np.array with 0-th column true ratings, and 1-th column estimated ratings
		predictions_df = predictions_dict[peer]
		if predictions_df.size != 0:
			# estimated_values is now 2D column vector
			estimated_values = np.array([predictions_df[:,1]]).transpose()

			user_item_pair_counter 		+= len(estimated_values)
			# count non-nan values
			prediction_possible_counter	+= len(estimated_values) - np.count_nonzero(np.isnan(estimated_values))
	if user_item_pair_counter > 0:
		overall_coverage = prediction_possible_counter/user_item_pair_counter
	else:
		overall_coverage = np.nan
	return overall_coverage


def get_per_user_coverage(predictions_dict):
	""" For a predictions_dict, calculate the mean of the per-user coverages, where a user's coverage is the fraction
		of test items for which an estimated rating can be predicted (i.e. the estimated rating is not nan).
	Input:
		- predictions_dict: <dict> of the forma: {userId: 2D np.array([[true_rating, estimated_rating]])
	Output:
		- per_user_coverage: <float> between 0 and 1 that represents the percentage of user-item pairs of the test set
							that allow rating prediction
	"""
	user_coverages = []

	for peer in predictions_dict.keys():
		# predictions_df is a 2D np.array with 0-th column true ratings, and 1-th column estimated ratings
		predictions_df = predictions_dict[peer]
		if predictions_df.size != 0:
			# estimated_values is now 2D column vector
			estimated_values = np.array([predictions_df[:,1]]).transpose()

			user_item_pair_counter 		= len(estimated_values)
			# count non-nan values
			prediction_possible_counter	= len(estimated_values) - np.count_nonzero(np.isnan(estimated_values))
			user_coverage				= prediction_possible_counter/user_item_pair_counter
			user_coverages.append(user_coverage)

	if len(user_coverages) > 0:
		per_user_coverage = sum(user_coverages)/len(user_coverages)
	else:
		per_user_coverage = np.nan
	return per_user_coverage





def get_user_space_coverage(snapshot_output_path, peers, k_items = 10, drop_percentile = None, by = "sim_to_sender", k_min = 1):
	""" For a predictions_dict, calculate the utility coverage as the number of users that can predict (! not necessarily recommend) at least
	k_items to themselves.

	Input:
		- predictions_dict: <dict> of the forma: {userId: 2D np.array([[true_rating, estimated_rating]])
	Output:
		- user_space_coverage: <float> between 0 and 1 that represents the percentage of user that can predict at least k_items.
	"""
	counter = 0

	for peer in peers:
		# load the peer_snapshot_df
		snapshot_df = pd.read_csv(os.path.join(snapshot_output_path, peer+".csv"), index_col = 0, dtype = {'userId': str, 'itemId': str} )
		# DROP PERCENTILE (of BY)
		# only keep those entries of the snapshot_df that are within the drop_percentile (in the <by> category)
		################################################################################################
		if drop_percentile is not None:
			snapshot_df = drop_entries(snapshot_df, drop_percentile = drop_percentile, by = by)

		# k_min: ONLY CONSIDER ITEMS FOR WHICH AT LEAST k_min RATINGS ARE AVAILABLE
		################################################################################################
		itemId_col = snapshot_df.loc[:,"itemId"] # itemId_col is a pandas.Series
		itemId_counts = itemId_col.value_counts()
		# drop all items (with index itemId) that have less than k_min entries
		itemId_counts = itemId_counts[itemId_counts>= k_min]

		# check if there are entries for at least k_items unique items
		num_unq_items = len(itemId_counts)#itemId_col.drop_duplicates() #snapshot_df.loc[:,"itemId"].drop_duplicates()
		# if there are at least k_items unique items, increment the counter
		if num_unq_items>=k_items:
			counter += 1
	if len(peers) > 0:
		user_space_coverage = counter/len(peers)
	else:
		user_space_coverage = np.nan
	return user_space_coverage


def get_utility_coverage(predictions_dict, k_items = 10, threshold = 3.5):
	""" For a predictions_dict, calculate the utility coverage as the number of users that are recommended (= estimated rating >= threshold) at least
	k_items.

	Input:
		- predictions_dict: <dict> of the forma: {userId: 2D np.array([[true_rating, estimated_rating]])
	Output:
		- utility_coverage: <float> between 0 and 1 that represents the percentage of user that are recommended at least k_items .
	"""
	counter = 0

	for peer in predictions_dict.keys():
		# predictions_df is a 2D np.array with 0-th column true ratings, and 1-th column estimated ratings
		predictions_df = predictions_dict[peer]
		if predictions_df.size != 0:
			# estimated_values is now 2D column vector
			estimated_values = np.array([predictions_df[:,1]]).transpose()
			# handle nan values in the estimated_values
			estimated_values = drop_nan(estimated_values)
			# count the number of recommended items (to peer)
			recommended_items_count = np.count_nonzero(estimated_values >= threshold)
			if recommended_items_count >= k_items:
				counter += 1
	dict_len = len(list(predictions_dict.keys()))
	if 	dict_len == 0:
		utility_coverage = 0.0
	else:
		utility_coverage = counter / len(list(predictions_dict.keys()))

	return utility_coverage



def aggregated_CFformula(item_snapshot_df):
	""" For a rating dataframe of a single "userId" and "itemId" yet possibly a larger set of "sender"s,
		with columns : ["sender", "userId", "itemId", "rating", "sim","agg_sim", "sim_to_sender"]

		apply the CF formula as

		dot-product("rating"-column, "agg_sim"-column) / sum("agg_sim"-column)
	"""
	# pick only the entries with positive "aggregated similarities"
	pos_sim_item_snapshot_df = item_snapshot_df[item_snapshot_df["agg_sim"] >= 0.0]
	# calculate the denominator
	denominator = pos_sim_item_snapshot_df.loc[:,"agg_sim"].sum()
	if denominator == 0.0:
		return np.nan
	else:
		numerator = np.dot(pos_sim_item_snapshot_df.loc[:,"rating"], pos_sim_item_snapshot_df.loc[:,"agg_sim"])
		return numerator/denominator

def vanilla_CFformula_with_mean_sims(item_snapshot_df):
	""" For a rating dataframe of many "userIds" that are the output of the DecCF algorithm, with columns

	apply the CF formula (weighted average). Ratings weighted by similarity,
	where the similarity is formaed by the average of "sim_to_sender" and "sim"
	"""
	# pick only the entries with positive similarities
	pos_sim_item_snapshot_df = item_snapshot_df[item_snapshot_df["sim_to_sender"] >= 0.0]

	av_sims = (pos_sim_item_snapshot_df.loc[:,"agg_sim"] + pos_sim_item_snapshot_df.loc[:,"sim_to_sender"])/2

	# calculate the denominator
	denominator = (av_sims).sum()
	if denominator == 0.0:
		return np.nan
	else:
		numerator = np.dot(pos_sim_item_snapshot_df.loc[:,"rating"], av_sims)
		return numerator/denominator

def vanilla_CFformula_with_sim_to_sender(item_snapshot_df):
	""" For a rating dataframe of many "userIds" that are the output of the DecCF algorithm, with columns

	apply the CF formula (weighted average). Ratings weighted by similarity, by substituting "sim" with "agg_sim". (cf. <vanilla_CFformula>

	"""
	# pick only the entries with positive similarities
	pos_sim_item_snapshot_df = item_snapshot_df[item_snapshot_df["sim_to_sender"] >= 0.0]
	# calculate the denominator
	denominator = pos_sim_item_snapshot_df.loc[:,"sim_to_sender"].sum()
	if denominator == 0.0:
		return np.nan
	else:
		numerator = np.dot(pos_sim_item_snapshot_df.loc[:,"rating"], pos_sim_item_snapshot_df.loc[:,"sim_to_sender"])
		return numerator/denominator

def vanilla_CFformula(item_snapshot_df):
	""" For a rating dataframe of many "userIds" that are the output of the DecCF algorithm, with columns

	apply the CF formula (weighted average). Ratings weighted by similarity.

	"""
	# pick only the entries with positive similarities
	pos_sim_item_snapshot_df = item_snapshot_df[item_snapshot_df["sim"] >= 0.0]
	# calculate the denominator
	denominator = pos_sim_item_snapshot_df.loc[:,"sim"].sum()
	if denominator == 0.0:
		return np.nan
	else:
		numerator = np.dot(pos_sim_item_snapshot_df.loc[:,"rating"], pos_sim_item_snapshot_df.loc[:,"sim"])
		return numerator/denominator


def unweighted_CFformula(item_snapshot_df):
	""" For a rating dataframe of many "userIds" that are the output of the DecCF algorithm, with columns

	apply the unweighted CF formula.

	"""
	# pick only the entries with positive similarities
	pos_sim_item_snapshot_df = item_snapshot_df[item_snapshot_df["sim"] >= 0.0]
	# calculate the denominator
	denominator = pos_sim_item_snapshot_df.shape[0]
	if denominator == 0.0:
		return np.nan
	else:
		return np.mean(pos_sim_item_snapshot_df.loc[:,"rating"])


def flatten(list_of_lists):
	return list(chain.from_iterable(list_of_lists))




def make_predictions_dict_entry_neighborhood_version(tuple):
	""" Calculate a predictions dict entry for a given peer (and snapshots_output_path, and test_df.
	Since the <map> function of the Pool class only supports one argument, we have to bundle them into triples.
	Mainly used for the parallel computation of <make_predictions_dict_inparallel>.
	Input:
		- tuple : tuple of parameters of the format

					peer                 = tuple[0]
					snapshot_output_path = tuple[1]
					test_df              = tuple[2]
					CFformula			 = tuple[3]
					drop_nans			 = tuple[4]
					k_min				 = tuple[5]
					k_max				 = tuple[6]
					by					 = tuple[7] : default "sim_to_sender"
		- by	: <str> with a column to sort (descending) the ratings in the snapshot for slicing
	Output:
		- {userId : [[true_rating, estimated_rating],..]}
	"""
	peer                 = tuple[0]
	snapshot_output_path = tuple[1]
	test_df              = tuple[2]
	CFformula			 = tuple[3]
	drop_nans			 = tuple[4]
	k_min				 = tuple[5]
	k_max				 = tuple[6]
	by					 = tuple[7]

	predictions_dict_entry = []
	# select the peers test_df slice
	peer_test_df = test_df[test_df["userId"] == peer]
	# pick the unq items in peer_test_df
	unq_peer_items = peer_test_df.loc[:, "itemId"].drop_duplicates()
	# if the unq peer's items are not empty
	if unq_peer_items.size !=0:
		# load the peer_snapshot_df
		snapshot_df = pd.read_csv(os.path.join(snapshot_output_path, peer+".csv"), index_col = 0)
		# for all items in peer_test_df
		for ind in peer_test_df.index:
			itemId           = peer_test_df.loc[ind,"itemId"]
			item_snapshot_df = snapshot_df[snapshot_df["itemId"] == itemId]

			if (k_min != -1) & (item_snapshot_df.shape[0] < k_min):
				continue
			if (k_max != -1) & (item_snapshot_df.shape[0] > k_max):
				item_snapshot_df = item_snapshot_df.sort_values(by = by, ascending = False)[:k_max]

			estimated_rating = CFformula(item_snapshot_df)
			true_rating      = peer_test_df.loc[ind, "rating"]
			# append [true_rating, estimated_rating] to predictions_dict_entry
			# if estimated_rating = np.nan is to be dropped
			if drop_nans:
				# if estimated_rating == np.nan
				if np.isnan(estimated_rating):
					# do nothing
					continue
				else:# append [true_rating, estimated_rating]
					predictions_dict_entry.append([true_rating, estimated_rating])
			else: #append
				predictions_dict_entry.append([true_rating, estimated_rating])


			# append [true_rating, estimated_rating] to predictions_dict_entry
			#if drop_nans & ~ np.isnan(estimated_rating):
			#	predictions_dict_entry.append([true_rating, estimated_rating])

		predictions_dict_entry = np.array(predictions_dict_entry)
	else:
		predictions_dict_entry = np.array([])
	return predictions_dict_entry


def map_Ids(list_of_Ids):
	""" Map all user/itemIds (ordered) from
	Input:
		- userIds [unique]
	Output:
		- self.__userId_dict : <dict> of the format {userId: index in similarity_matrix}
													keys are arbitrary: values are <int> !!!
	"""
	Id_dict = {Id : i for i, Id in enumerate(list_of_Ids,0)}

	return Id_dict



def coord_to_csr_matrix(coord_matrix, userId_dict, itemId_dict, nrows = None, ncols = None):
	""" Tranform a user/neighborhood-item-matrix in the coordinate form with row and col identifiers ["userId"] and ["itemId"] and values in ["rating"]
	into a dense matrix format.
	Input:
		- coord_matrix	:
		- userId_dict	:
		- itemId_dict	:
	Output:
		- dense_matrix	:


	"""
	row  = coord_matrix.loc[:,"userId"].values#astype(str).values
	col  = coord_matrix.loc[:,"itemId"].values#stype(str).values
	# the row and col identifiers (that can be any type e.g. <str> or <float>) have to be mapped to type <int>
	row  = [userId_dict[userId] for userId in row]
	col  = [itemId_dict[itemId] for itemId in col]

	# in order to row and col, we associate the ratings as np.float64
	data = coord_matrix.loc[:,"rating"].astype(np.float64).values

	# create scipy coordinate matrix
	if (nrows is not None) & (ncols is not None):
		D_coord		= coo_matrix((data, (row, col)), shape=(nrows, ncols))
	else:
		D_coord		= coo_matrix((data, (row, col)))
	# cast to sparse
	D_sparse	= csr_matrix(D_coord)

	return D_sparse


def vanilla_CFformula_nparray_version(sim_rating_array):
	""" For a 2D array of shape (n,2) with similarities in the first columns and ratings in the second.
	calculate the vanilla user-based CF weighted average formula. """

	denominator = sim_rating_array[:,0].sum()

	if denominator != 0.0:
		#                           sim_column * rating_column
		numerator = (sim_rating_array[:,0]*sim_rating_array[:,1]).sum()
		return numerator/denominator
	else:
		return np.nan


"""
def make_predictions_dict_entry_neighborhood_version(parameter_list):

	# variable
	peer							= parameter_list[0]
	# fixed
	test_df							= parameter_list[1]
	neighborhood_item_matrix_coord	= parameter_list[2]
	itemId_dict						= parameter_list[3]
	userId_dict						= parameter_list[4]
	neighborhoodId_dict				= parameter_list[5]
	pairwise_sims					= parameter_list[6]
	k_min							= parameter_list[7]
	k_max							= parameter_list[8]
	neighborhood_item_matrix_sparse	= parameter_list[9]

	# the peer index is the row-index in user_item_matrix_sparse that corresponds to peer
	peer_index		= userId_dict[peer]
	#	Query the user-profile of peer from train_df
	peer_test_df	= test_df[test_df["userId"] == peer]

	# initialize the result list
	predictions_dict_entry = []

	unq_peer_items = peer_test_df.loc[:, "itemId"].drop_duplicates()
	# if the unq peer's items are not empty
	if unq_peer_items.size !=0:
		for ind in peer_test_df.index:
			true_rating		= peer_test_df.loc[ind, "rating"]
			item			= peer_test_df.loc[ind, "itemId"]
			# the item index is the column-index in neighborhood_item_matrix_sparse (!) NOT user_item_matrix_sparse that corresponds to item
			item_index = itemId_dict[item]
			#	Run user-based CF on the neigh. user-item matrix
			#	0. Find all users that have rated the item
			candidate_neighborhoods = neighborhood_item_matrix_coord[neighborhood_item_matrix_coord["itemId"] == item].loc[:,"userId"].drop_duplicates().tolist()
			#	1. Find the k_max (all if k_max == -1) users' that have rated item
			if candidate_neighborhoods != []:
				candidate_neighborhood_indices	= [neighborhoodId_dict[candidate_neighborhood] for candidate_neighborhood in candidate_neighborhoods]
				candidate_sims					= pairwise_sims[peer_index, candidate_neighborhood_indices]
				candidate_sims_and_neigh_indices= [el for el in zip(candidate_sims, candidate_neighborhood_indices)]
				# sorted_candidate_sims_and_neigh_indices :
				# e.g. [(0.6042374377598074, 4), (0.5594023668122983, 5), (0.5557824223928495, 2)]
				sorted_candidate_sims_and_neigh_indices = sorted(candidate_sims_and_neigh_indices, key=lambda x: x[0], reverse = True)

				# if there are too few ratings to form the weighted average over, then continue with the next loop iteration
				if (k_min != -1) & (len(sorted_candidate_sims_and_neigh_indices) < k_min):
					continue
				# if there are too many (>k_max) (neighborhoods') ratings to consider, pick the k_max rating that correspond to the k_max most similar neighborhoods
				if (k_max != -1) & (len(sorted_candidate_sims_and_neigh_indices) > k_max):
					sorted_candidate_sims_and_neigh_indices = sorted_candidate_sims_and_neigh_indices[:k_max]

				# match the neighborhood indices with the respective ratings
				sims_and_ratings = np.array([[sim, neighborhood_item_matrix_sparse[neighborhood_index, item_index]] for sim, neighborhood_index in sorted_candidate_sims_and_neigh_indices])

				estimated_rating = vanilla_CFformula_nparray_version(sims_and_ratings)
			else:
				estimated_rating = np.nan

				# fill in an np.nan in combination with the item's true rating (with all the other assessments)

			predictions_dict_entry.append([true_rating, estimated_rating])

		# cast perdictions_dict_entry to numpy.array
		predictions_dict_entry = np.array(predictions_dict_entry)
	else:
		predictions_dict_entry = np.array([[]])
	return predictions_dict_entry
"""

def make_predictions_dict_neighborhood_version(train_df, test_df, neighborhood_item_matrix_path, k_min = -1, k_max = -1):
	""" For a train_df, and corresponding test_df (pandas.DataFrame coordinate matrix with columns "userId", "itemId", "rating"), and neighborhood_item_matrix_path.
	Calculate a predictions_dict, a dictionary that holds the true and estimated ratings of all ratings in the test_df. The predictions_dict holds estimates
	of user_profiles that run user-based CF on the abstract matrix of neighborhood profiles that have been generated by DecAggCF.py. (cf. <> for the generation of the
	abstract neighborhood - item matrix.

	Input:
		- train_df						: pandas.DataFrame with columns ["userId", "itemId", "ratings"] (e.g. by a train-test split as per <>)
		- test_df						: pandas.DataFrame with columns ["userId", "itemId", "ratings"]
		- neighborhood_item_matrix_path : <str> path to a neighborhood_item_matrix that holds aggregated neighborhood users
		- CFformula						: the CF formula has to be able to process the .csv files specified in snapshot_output_path for the calculation of the estimated rating
		- peers							:
		- drop_nans						:
		- k_min							:
		- k_max							:


	Output:
		- predictions_dict : <dict> of the format {peer: np.array([[true_rating (of userId), estimated_rating (of userId)]])}

	Example:
	>>> predictions_dict
	{'1': array([[2. , 3.5],
       [3. , 3. ]]), '2': array([[ 2., nan]]), '3': array([[1.        , 4.33333333]]), '4': array([[ 3., nan]]), '6': array([[5., 3.]]), '8': array([[ 2., nan]])}

	Interpretation: user '1' has duples of (true rating, estimated rating). Note that the item IDs are not saved.
	"""
	# LOAD NEIGHBORHOOD-ITEM-MATRIX
	####################################
	print("Load neighborhood-item-matrix..")
	neighborhood_item_matrix_coord = pd.read_csv(neighborhood_item_matrix_path, index_col = 0)
	# extract unique items
	unq_itemIds 		= pd.concat([train_df.loc[:,"itemId"], test_df.loc[:,"itemId"]], axis = 0).drop_duplicates().tolist()
	itemId_dict 		= map_Ids(unq_itemIds)
	# extract unique neighborhoods ("userId")
	unq_neighborhoodIds = neighborhood_item_matrix_coord.loc[:, "userId"].drop_duplicates().tolist()
	neighborhoodId_dict = map_Ids(unq_neighborhoodIds)
	# cast coord neighborhood_item_matrix to sparse-form (csr-format)
	neighborhood_item_matrix_sparse = coord_to_csr_matrix(neighborhood_item_matrix_coord, neighborhoodId_dict, itemId_dict,\
																						nrows = len(unq_neighborhoodIds), ncols = len(unq_itemIds))
	#

	# LOAD (train_df) USER-ITEM-MATRIX
	####################################
	# extract unique (train) users ("userId") :: here we have to guarantee that every test user is also reflected in the train_df, for else the algo is non-sensical
	unq_userIds = train_df.loc[:, "userId"].drop_duplicates().tolist()
	userId_dict = map_Ids(unq_userIds)
	# cast coord neighborhood_item_matrix to sparse-form (csr-format)
	user_item_matrix_sparse = coord_to_csr_matrix(train_df, userId_dict, itemId_dict, nrows = len(unq_userIds), ncols = len(unq_itemIds))

	# CALCULATE PAIRWISE SIMS BETWEEN USERS AND NEIGHBORHOODS
	##########################################################

	pairwise_sims = cosine_similarity(user_item_matrix_sparse, neighborhood_item_matrix_sparse, dense_output=True)

	# initialize predictions_dict [keys: userId, values: nx2 np.array with row-wise duples of true and estimated ratings (on unknown/untracked itemIds)]
	predictions_dict = dict()

	# FOR EVERY PEER IN test_df
	# NOTE that peers_in_test_df are identifiers rather than indices (as in userId_dict)
	peers_in_test_df = test_df.loc[:,"userId"].drop_duplicates()

	for peer in peers_in_test_df:
		# define parameter list
		## variable
		#peer							= parameter_list[0]
		## fixed
		#test_df							= parameter_list[1]
		#neighborhood_item_matrix_coord	= parameter_list[2]
		#itemId_dict						= parameter_list[3]
		#userId_dict						= parameter_list[4]
		#neighborhoodId_dict				= parameter_list[5]
		#pairwise_sims					= parameter_list[6]
		#k_min							= parameter_list[7]
		#k_max							= parameter_list[8]
		#neighborhood_item_matrix_sparse	= parameter_list[9]




		parameter_list = [peer, test_df, neighborhood_item_matrix_coord, itemId_dict, userId_dict, neighborhoodId_dict, pairwise_sims,
							k_min, k_max, neighborhood_item_matrix_sparse]

		# initialize predictions_dict_entry
		predictions_dict_entry = make_predictions_dict_entry_neighborhood_version(parameter_list)

		# append predictions dict entry
		predictions_dict[peer] = predictions_dict_entry

	return predictions_dict

def make_predictions_dict_neighborhood_version_inparallel(train_df, test_df, neighborhood_item_matrix_path, k_min = -1, k_max = -1):
	""" For a train_df, and corresponding test_df (pandas.DataFrame coordinate matrix with columns "userId", "itemId", "rating"), and neighborhood_item_matrix_path.
	Calculate a predictions_dict, a dictionary that holds the true and estimated ratings of all ratings in the test_df. The predictions_dict holds estimates
	of user_profiles that run user-based CF on the abstract matrix of neighborhood profiles that have been generated by DecAggCF.py. (cf. <> for the generation of the
	abstract neighborhood - item matrix.

	Input:
		- train_df						: pandas.DataFrame with columns ["userId", "itemId", "ratings"] (e.g. by a train-test split as per <>)
		- test_df						: pandas.DataFrame with columns ["userId", "itemId", "ratings"]
		- neighborhood_item_matrix_path : <str> path to a neighborhood_item_matrix that holds aggregated neighborhood users
		- CFformula						: the CF formula has to be able to process the .csv files specified in snapshot_output_path for the calculation of the estimated rating
		- peers							:
		- drop_nans						:
		- k_min							:
		- k_max							:


	Output:
		- predictions_dict : <dict> of the format {peer: np.array([[true_rating (of userId), estimated_rating (of userId)]])}

	Example:
	>>> predictions_dict
	{'1': array([[2. , 3.5],
       [3. , 3. ]]), '2': array([[ 2., nan]]), '3': array([[1.        , 4.33333333]]), '4': array([[ 3., nan]]), '6': array([[5., 3.]]), '8': array([[ 2., nan]])}

	Interpretation: user '1' has duples of (true rating, estimated rating). Note that the item IDs are not saved.
	"""
	# LOAD NEIGHBORHOOD-ITEM-MATRIX
	####################################
	print("Load neighborhood-item-matrix..")
	neighborhood_item_matrix_coord = pd.read_csv(neighborhood_item_matrix_path, index_col = 0)
	# extract unique items
	unq_itemIds 		= pd.concat([train_df.loc[:,"itemId"], test_df.loc[:,"itemId"]], axis = 0).drop_duplicates().tolist()
	itemId_dict 		= map_Ids(unq_itemIds)
	# extract unique neighborhoods ("userId")
	unq_neighborhoodIds = neighborhood_item_matrix_coord.loc[:, "userId"].drop_duplicates().tolist()
	neighborhoodId_dict = map_Ids(unq_neighborhoodIds)
	# cast coord neighborhood_item_matrix to sparse-form (csr-format)
	neighborhood_item_matrix_sparse = coord_to_csr_matrix(neighborhood_item_matrix_coord, neighborhoodId_dict, itemId_dict,\
																						nrows = len(unq_neighborhoodIds), ncols = len(unq_itemIds))
	#

	# LOAD (train_df) USER-ITEM-MATRIX
	####################################
	# extract unique (train) users ("userId") :: here we have to guarantee that every test user is also reflected in the train_df, for else the algo is non-sensical
	unq_userIds = train_df.loc[:, "userId"].drop_duplicates().tolist()
	userId_dict = map_Ids(unq_userIds)
	# cast coord neighborhood_item_matrix to sparse-form (csr-format)
	user_item_matrix_sparse = coord_to_csr_matrix(train_df, userId_dict, itemId_dict, nrows = len(unq_userIds), ncols = len(unq_itemIds))

	# CALCULATE PAIRWISE SIMS BETWEEN USERS AND NEIGHBORHOODS
	##########################################################

	pairwise_sims = cosine_similarity(user_item_matrix_sparse, neighborhood_item_matrix_sparse, dense_output=True)

	# DEFINE THE GRID OF PARAMETER COMBINATIONS FOR PARALLEL PROCESSING
	############
	peers_in_test_df = test_df.loc[:,"userId"].drop_duplicates().tolist()

	parameter_grid = product(peers_in_test_df, [test_df], [neighborhood_item_matrix_coord], [itemId_dict], [userId_dict], [neighborhoodId_dict], [pairwise_sims],
							[k_min], [k_max], [neighborhood_item_matrix_sparse])

	# initialize a results list
	predictions_dict_entries = []
	predictions_dict = dict()

	# start a pool of workers
	p = Pool()# automatically uses the max number of cpus available
	# initialize the previously calculated future snapshots
	# NOTE that the output of self.inner_for_loop is [] if there are no aggregated ratings to be passed, they have to be filtered out!
	results_object = p.map_async(make_predictions_dict_entry_neighborhood_version, parameter_grid, callback = predictions_dict_entries.append)

	# wait until all results have been calculated
	results_object.wait()

	# flatten results list
	predictions_dict_entries = flatten(predictions_dict_entries)

	# close and join the pool workers
	p.close()
	p.join()

	# collect results and put them into the dictionary
	for peer, predictions_dict_entry in zip(peers, predictions_dict_entries):
		predictions_dict[peer] = predictions_dict_entry


	return predictions_dict




def drop_rows_with_nans(np_array):
	""" Drop rows of a numpy.array that contain nan values. """
	np_array_without_nan_rows = np_array[~np.isnan(np_array).any(axis=1), :]
	return np_array_without_nan_rows

def remove_nan_rows_from_predictions_dict(predictions_dict_with_nans):
	""" Remove the rows in the values of a predictions_dict with nans. """
	predictions_dict_without_nans = dict()
	for key in predictions_dict_with_nans.keys():
		predictions_array_with_nans 	= predictions_dict_with_nans[key].copy()
		predictions_array_without_nans	= drop_rows_with_nans(predictions_array_with_nans)
		predictions_dict_without_nans[key] = predictions_array_without_nans

	return predictions_dict_without_nans


def make_baseline_predictions_dict(predictions_dict, mean_dict):
	""" Take a mean_dict and impute the mean rating of every user into the predicted ratings column (index = 1).


		{'user1': array([[5.,nan],		       	{'user1': array([[5.,4.4125],
		 		 	     [4.,4.85123]])} -->                     [4.,4.4125]])}

		where the average rating by 'user1' in the train_df is 4.4125.

	"""
	baseline_predictions_dict = dict()

	for key in predictions_dict.keys():
		predictions_array = predictions_dict[key].copy()
		predictions_array[:,1] = mean_dict[key]
		baseline_predictions_dict[key] = predictions_array

	return baseline_predictions_dict




def make_mean_dict_entry(rating_df, user):
	""" Make a dictionary entry that is the mean of the ratings in the <rating_df> by <user>.

		Input:
			- rating_df: <pandas.DataFrame>
			- user: <string>

		Output:
			- mean_dict: <dict>
	"""
	user_ratings = rating_df[rating_df.loc[:,'userId']==user].loc[:,'rating']
	mean_user_rating = user_ratings.mean()

	return mean_user_rating


def make_mean_dict(rating_df):
	""" Make a dictionary of the format: {'userId':mean_user_rating}. This dictionary will be used to
		apply a mean-centered version of the user-based CF formula.

		Input:
			- rating_df: <pandas.DataFrame>

		Output:
			- mean_dict: <dict>
	"""
	users = rating_df.loc[:,'userId'].drop_duplicates()
	mean_dict = {user:make_mean_dict_entry(rating_df, user) for user in users}

	return mean_dict




def subtract_mean_rating(rating_df, mean_dict):
	""" Subtract the mean rating of a user specified in the keys of the <mean_dict> from all ratings of a specific user in the <rating_df>.
		Input:
			-
			-
		Output:
			-
	"""
	# make a mean_rating column that can be subtracted from the 'rating' column

	mean_rating_col = rating_df.loc[:,'userId'].copy().apply(lambda x: mean_dict[x])

	rating_df.loc[:,'rating'] -= mean_rating_col

	return rating_df


def make_predictions_dict_inparallel(test_df, snapshot_output_path = "../data/snapshots/default_snapshot_output_path",\
								CFformula = aggregated_CFformula, peers = None, drop_nans = True, k_min = -1, k_max = -1, by ="sim_to_sender",\
								drop_percentile = None, timestamp_min = None, mean_dict = None):
	""" The same as <make_predictions_dict> yet with parallel processes for speedup.
	Create a predictions_dict <dict> that holds the test entries paired with the estimated ratings (=predicted ratings = predictions).

	Format:
	#######
	>>> predictions_dict
	{<userId>: array([[<true_rating(userId0, itemId0,0)>, <estimated_rating(userId0, itemId0,0)>], [<true_rating(userId0, itemId0,1)>, <estimated_rating(userId0, itemId0,1)>], ...]),
					 [[<true_rating(userId1, itemId1,0)>, <estimated_rating(userId1, itemId1,0)>], [<true_rating(userId1, itemId1,1)>, <estimated_rating(userId1, itemId1,1)>], ...]),
						...}

	Example:
	########
 	>>> predictions_dict
 	{'1': array([[2. , 3.5],
        [3. , 3. ]]), '2': array([[ 2., nan]]), '3': array([[1.        , 4.33333333]]), '4': array([[ 3., nan]]), '6': array([[5., 3.]]), '8': array([[ 2., nan]])}


	Input:
	######
		- test_df 				: <pandas.DataFrame> that holds the test entries of the user-item matrix in coordinate format, i.e. columns ['userId', 'itemId', 'rating', ['timestamp']].
								  It is for instance the output of the <utilities.util.make_train_test_split> function.
		- snapshot_output_path	: <str> directory that holds the snapshot files. Recall that a snapshot file is a <userId>.csv with the columns ['userId', 'itemId', 'rating', 'sim,agg_sim', 'sender', 'sim_to_sender', 'timestamp'].
								  Every snapshot file represents the ratings that the user with <userId> has collected.]
		- CFformula				: function reference to a user-based collaborative filtering formula. See for instance <evaluation.evaluation.vanilla_CFformula>.
		- peers 				:
		- drop_nans				: <Boolean>; if True, drop all entries in the predictions_dict that holds NAN values in the prediction column. Else, do nothing. See details on Output.
		- k_min					: <Int> only consider estimated ratings, if there are at least <k_min> ratings available to use with <CFformula>.
		- k_max					: <Int> only consider the top <k_max> ratings per estimated rating measured with respect to <by>. For instance, if <by> = "sim", then only the <k_max> ratings with the highest
		 						  "sim" values are considered for rating estimation.
		- by					:
		- drop_percentile		: <float> drop all entries of a snapshot_df that are within the drop_percentile of <by>
								 e.g. by = "sim_to_sender" and drop_percentile = 0.9 --> all entries of snapshot_df that are within the
								 lower 90% of the "sim_to_sender" distribution. Afterwards estimate ratings on the rest
		- timestamp_min			: <int> timestamp threshold. older entries are dropped before being considered for rating estimation.
		- mean_dict				: <dict>; mean dictionary with key:value pairs of <userId>:<mean_rating_of_userId>. This is for instance the output of the <evaluation.evaluation.make_mean_dict> function.
	"""

	# initialize a results list
	predictions_dict_entries = []
	predictions_dict = dict()
	# extract all unique userIds from test_df
	peers = test_df["userId"].drop_duplicates()
	# start a pool of workers


	# start a worker pool
	with Pool() as pool:# automatically uses the max number of cpus available


		results_object = pool.map_async(make_predictions_dict_entry, product(peers, [snapshot_output_path], [test_df], [CFformula], [drop_nans],\
																				[k_min], [k_max], [by], [drop_percentile], [timestamp_min], [mean_dict]))
		results_object.wait()

		predictions_dict_entries = results_object.get()


	for peer, predictions_dict_entry in zip(peers, predictions_dict_entries):
		predictions_dict[peer] = predictions_dict_entry

	return predictions_dict




def make_predictions_dict_entry(tuple):
	""" Calculate an entry (that is a key:value pair) for the <predictions_dict>.
	Since the <map> function of the Pool class only supports one argument, we have to bundle them into triples.
	Mainly used for the parallel computation of <make_predictions_dict_inparallel>.
	Input:
		- tuple : tuple of parameters of the format (see <evaluation.evaluation.make_predictions_dict_inparallel> for details)

					peer                 = tuple[0]
					snapshot_output_path = tuple[1]
					test_df              = tuple[2]
					CFformula			 = tuple[3]
					drop_nans			 = tuple[4]
					k_min				 = tuple[5]
					k_max				 = tuple[6]
					by					 = tuple[7]
					drop_percentile		 = tuple[8]
					timestamp_min		 = tuple[9]
					mean_dict			 = tuple[10]

	Output:
		- [[true_rating, estimated_rating],..]
	"""
	peer                 = tuple[0]
	snapshot_output_path = tuple[1]
	test_df              = tuple[2]
	CFformula			 = tuple[3]
	drop_nans			 = tuple[4]
	k_min				 = tuple[5]
	k_max				 = tuple[6]
	by					 = tuple[7]
	drop_percentile		 = tuple[8]
	timestamp_min		 = tuple[9]
	mean_dict			 = tuple[10]

	# initialize the entry
	predictions_dict_entry = []
	# select the peer's test_df slice
	peer_test_df = test_df[test_df["userId"] == peer]
	# pick the unq items in peer_test_df #TODO: Check if obsolete to drop_duplicates
	unq_peer_items = peer_test_df.loc[:, "itemId"].drop_duplicates()
	# if the unq peer's items are not empty, that is there are test entries associated with <userId>
	if unq_peer_items.size !=0:
		# load the peer_snapshot_df
		snapshot_df = load_snapshot_df(snapshot_output_path, peer)

		# [OPTIONAL] DROP OLDER THAN timestamp_min
		###########################################################
		if timestamp_min is not None:
			snapshot_df = snapshot_df[snapshot_df["timestamp"]>= timestamp_min]

		# [OPTIONAL] DROP PERCENTILE (of BY)
		# only keep those entries of the snapshot_df that are within the drop_percentile (in the <by> category)
		################################################################################################
		if drop_percentile is not None:
			snapshot_df = drop_entries(snapshot_df, drop_percentile = drop_percentile, by = by)

		# [OPTIONAL] SUBTRACT the mean ratings from the ratings in the snapshot_df
		###########################################################
		if mean_dict is not None:
			snapshot_df = subtract_mean_rating(snapshot_df, mean_dict)

		# for all items in <peer_test_df> (in test_df all itemIds are unique for a single userId)
		for ind in peer_test_df.index:

			# select its corresponding entries in <snapshot_df>
			itemId           = peer_test_df.loc[ind,"itemId"]
			item_snapshot_df = snapshot_df[snapshot_df["itemId"] == itemId]
			# !!!! item_snapshot_df ist immer noch empty!
			# and its true rating
			true_rating      = peer_test_df.loc[ind, "rating"]
			# if there are less than k_min entries available
			if (k_min != -1) & (item_snapshot_df.shape[0] < k_min):
				# just continue with the next item if drop_nans is set
				if drop_nans:
					continue
				# else set the estimated rating as np.nan
				else:
					estimated_rating = np.nan
			# else if there are enough entries
			else:
				# check whether there are too many of them, if yet, prune the entries (by the top "by" category)
				if (k_max != -1) & (item_snapshot_df.shape[0] > k_max):
					item_snapshot_df = item_snapshot_df.sort_values(by = by, ascending = False)[:k_max]
				# then calculate the estimated rating according to the specified CFformula
				estimated_rating = CFformula(item_snapshot_df)

				if mean_dict is not None:
					estimated_rating += mean_dict[peer]

			# finally append the pair [true_rating, estimated_rating] to predictions_dict_entry
			predictions_dict_entry.append([true_rating, estimated_rating])
		# collect all [true_rating, estimated_rating] pairs for a userID to for man entry in the predictions_dict
		predictions_dict_entry = np.array(predictions_dict_entry)
	# in this case there are not test entries to estimate for that peer
	else:
		predictions_dict_entry = np.array([])
	return predictions_dict_entry




def make_predictions_dict(test_df, snapshot_output_path = "../data/snapshots/default_snapshots", \
											CFformula = aggregated_CFformula, peers = None, drop_nans = False, k_min = -1, k_max = -1,\
											by = "sim_to_sender", drop_percentile = None, timestamp_min = None, mean_dict = None):
	""" For a test_df (pandas.DataFrame coordinate matrix with columns "userId", "itemId", "rating"), and snapshot .csv files
	in a given snapshot_output_path (as e.g. produced by algo.DecentralizedAggregatingCF.save_future_snapshots) calculate a predictions dictionary
	that serves as a pre-product of accuracy, coverage, or ranked retrieval evaluation metrics.

	Input:
		- test_df              : pandas.DataFrame with columns ["userId", "itemId", "ratings"]
		- snapshot_output_path : <str> path to a directory that contains snapshots of the format <peer>.csv for every userId (=peer) in test_df
		- CFformula            : the CF formula has to be able to process the .csv files specified in snapshot_output_path for the calculation of the estimated rating
		- drop_percentile	   :
		- timestamp_min		   : <int> timestamp before which entries are not considered for rating estimation
	Output:
		- predictions_dict : <dict> of the format {peer: np.array([[true_rating (of userId), estimated_rating (of userId)]])}

	Example:
	>>> predictions_dict
	{'1': array([[2. , 3.5],
       [3. , 3. ]]), '2': array([[ 2., nan]]), '3': array([[1.        , 4.33333333]]), '4': array([[ 3., nan]]), '6': array([[5., 3.]]), '8': array([[ 2., nan]])}
	"""
	predictions_dict = dict()

	# extract all unique userIds from test_df
	peers = test_df["userId"].drop_duplicates()
	# NOTE that peer is of <str> type
	for peer in peers:
		tuple = (peer, snapshot_output_path, test_df, CFformula,drop_nans,\
				k_min, k_max, by, drop_percentile, timestamp_min, mean_dict)
		predictions_dict_entry = make_predictions_dict_entry(tuple)

		predictions_dict[peer] = predictions_dict_entry

	return predictions_dict

def get_accuracy_overall(predictions_dict):
	""" On the basis of the predictions_dict (as the output of <make_predictions_dict>, we calculate the overall accuracy simultaneously over all peers.
	Input:
		- predictions_dict	: <dict> (as the output of <make_predictions_dict>) of the format : {userId: np.array([[true_rating, estimated_rating]])}
	Output:
		- accuracy_dict		: <dict> a dictionary with the userId-wise accuracy scores of the format: {userId: np.float(accuracy)}
		- accuracy		: <np.float> overall accuracy score
	"""
	accuracy_dict = dict()
	# initialize the differences between true and (non-nan) estimated ratings
	all_true_and_estimated_ratings_df = None
	# drop nan values NOTE that only in the 1-th column (representing estimated ratings) NaNs can appear
	for key, value in predictions_dict.items():
		peer                = key
		predictions_nparray = value

		if predictions_nparray.size != 0:
			# handle nan values in the estimated_ratings
			predictions_nparray= drop_nan(predictions_nparray)
			if all_true_and_estimated_ratings_df is None:
				all_true_and_estimated_ratings_df = predictions_nparray
			else:
				all_true_and_estimated_ratings_df = np.concatenate([all_true_and_estimated_ratings_df, predictions_nparray], axis = 0)
			#differences.extend(list(true_ratings-estimated_ratings))

	if all_true_and_estimated_ratings_df is None:
		print("Predictions dictionary is empty!")
		return np.nan
	if all_true_and_estimated_ratings_df.shape[0] != 0:
		# unpack true and estimated ratings
		true_ratings      = all_true_and_estimated_ratings_df[:,0]
		estimated_ratings = all_true_and_estimated_ratings_df[:,1]

		MSE_overall = mean_squared_error(true_ratings, estimated_ratings)
	else:
		return np.nan

	return MSE_overall





def get_accuracy(predictions_dict):
	""" On the basis of a predictions_dict (as the output of <make_predictions_dict>, we calculate the accuracy scores per peer, and also take the mean over them.
	Input:
		- predictions_dict	: <dict> (as the output of <make_predictions_dict>) of the format : {userId: np.array([[true_rating, estimated_rating]])}
	Output:
		- accuracy_dict		: <dict> a dictionary with the userId-wise accuracy scores of the format: {userId: np.float(accuracy)}
		- mean_accuracy		: <np.float> mean over the userId-wise accuracy scores
	"""
	accuracy_dict = dict()
	MSEs = []
	# drop nan values NOTE that only in the 1-th column (representing estimated ratings) NaNs can appear
	for key, value in predictions_dict.items():
		peer                = key
		predictions_nparray = value

		if predictions_nparray.size != 0:
			# handle nan values in the estimated_ratings
			predictions_nparray= drop_nan(predictions_nparray)
		if predictions_nparray.size != 0:
			# unpack true and estimated ratings
			true_ratings      = predictions_nparray[:,0]
			estimated_ratings = predictions_nparray[:,1]

			MSE = mean_squared_error(true_ratings, estimated_ratings)
			MSEs.append(MSE)
		else:
			MSE = np.nan

		# add accuracy_dict
		accuracy_dict[peer] = MSE
	if len(MSEs) >0:
		# take the mean over the MSE accuracies
		mean_MSE = np.array(MSEs).mean()
	else:
		mean_MSE = np.nan
	return accuracy_dict, mean_MSE





def get_precision_and_recall(predictions_dict, threshold = 3.5, K = 10):
	""" For a predictions dictionary, assess the precision as the percentage of correctly recommended items
	to all users, where any item is recommended when the estimated rating is above the threshold. Any recommendation
	list (though) has a maximum of K items to recommend at max. Also calculate the recall as the fraction of
	correctly recommended items in relation to all 'relevant' items.

	WARNING: If this function required a predictions_dict where no NaNs (in case there are any) are dropped, for else
			 the number of relevant items as well as both relevant and recommended items cannot be assessed.

	Input:
		- predictions_dict	: <dict> output of
		- threshold			: <float>, items with estimated ratings above this threshold are recommended
		- K					: <int>, maximum length of the recommender list to a single user
	Output:
		- precisions_dict	: <dict> of precision values per "userId"
		- recalls_dict		: <dict> of recall values per "userId"

	"""
	precisions_dict = dict()
	recalls_dict    = dict()
	F1_dict			= dict()

	# peer is of type str
	for peer, user_predictions in predictions_dict.items():
		# user_predictions is a 2D-numpy array! 0-th column true_ratings, 1-th column estimated_ratings
		# sort predictions (of userId) by estimated rating [1-th column], NOTE that numpy.sort is not what we are looking for, since we want to sort rows instead of
		# within rows or within columns
		if user_predictions.size != 0:
			# number of relevant items [true_rating >= threshold]
			number_of_relevant_items = user_predictions[user_predictions[:,0] >= threshold].shape[0]
			# sort by estimated_rating (col = 1)
			sorted_user_predictions = np.array(sorted(user_predictions.tolist(), key = lambda x: x[1], reverse = True))
			# number of recommended items in top K
			number_of_recommended_items_in_topK = sorted_user_predictions[:K,:][sorted_user_predictions[:K,:][:,1] >= threshold].shape[0]
			# number of relevant [true_rating >= threshold (col = 0)] and recommended [estimated_rating >= threshold (col = 1)] items in top K
			number_of_relevant_and_recommended_items_in_topK = user_predictions[:K,:][ (user_predictions[:K,:][:,0] >= threshold) & (user_predictions[:K,:][:,1] >= threshold)].shape[0]
		else: # if no items are neither recommended nor relevant (keep in mind that no nans (rows) must be dropped for this function to work properly (cf. docstring))
			number_of_relevant_items 							= 0
			number_of_recommended_items_in_topK					= 0
			number_of_relevant_and_recommended_items_in_topK	= 0

		# Precision@K: Proportion of recommended items that are relevant
		# the precision trivially defaults to 1.0, when the K-lengthed recommendation list (number_of_recommended_items_in_topK) is empty
		precision_atK = number_of_relevant_and_recommended_items_in_topK / number_of_recommended_items_in_topK if number_of_recommended_items_in_topK != 0 else 1
		precisions_dict[peer] = precision_atK

		# Recall@K: Proportion of relevant items that are recommended
		recall_atK    = number_of_relevant_and_recommended_items_in_topK / number_of_relevant_items if number_of_relevant_items != 0 else 1
		recalls_dict[peer] = recall_atK

		# F1@K: F1 Statistic for every peer
		if recall_atK + precision_atK == 0:
			F1_atK = 0.0
		else:
			F1_atK		  = (2* precision_atK * recall_atK)/(recall_atK + precision_atK)

		F1_dict[peer] = F1_atK


	# calculate mean precision
	mean_precision	= safe_mean(np.array(list(precisions_dict.values())))
	# calculate mean recall
	mean_recall		= safe_mean(np.array(list(recalls_dict.values())))
	# F1 score
	mean_F1			= safe_mean(np.array(list(F1_dict.values())))

	return precisions_dict, recalls_dict, F1_dict, mean_precision, mean_recall, mean_F1



def collect_all_children(graph, peer, t = None):
	""" Collect all (!) children of a peer in a graph (not necessarily filled) recursively.
	E.g.
	1:0  2:0  3:0  4:0
	 |  / |  / |    |
	1:1  2:1  3:1  4:1
	 |  / |    |    |
	1:2  2:2  3:2  4:2

	collect_all_children(graph, "1", t = 2)
	>>> ['2', '3']

	"""
	if t == None:
		t = graph.T()

	if t == 1:
		leaf_children = set([el.split(":")[0] for el in graph.children(peer+":"+str(t))])
		return leaf_children

	else:
		all_children = set()
		children_at_time_prev_t = set([el.split(":")[0] for el in graph.children(peer+":"+str(t))])
		all_children            = all_children.union(children_at_time_prev_t)
		for child in children_at_time_prev_t:
			# perhaps I do not need to copy the result here
			result       = collect_all_children(graph, child, t = t-1)#.copy() set operations are (to the best of my knowledge) not in-place
			all_children = all_children.union(result)
		return all_children



def main():
	pickle_path = "../data/pickle_files/dataset=ratings.csvp=0.003T=3N=3fold_nr=1n_splits=2test_size=0.25random_state=426random_seed=51423.pk"
	pickle_path1 = "../data/pickle_files/test/first_dataset=ratings.csvp=0.003T=3N=3fold_nr=0n_splits=2test_size=0.25random_state=426random_seed=51423.pk"
	pickle_path2 = "../data/pickle_files/test/second_dataset=ratings.csvp=0.003T=3N=3fold_nr=0n_splits=2test_size=0.25random_state=426random_seed=51423.pk"
	pickle_path3 = "../data/pickle_files/dataset=ratings.csvp=0.001T=20N=3fold_nr=3n_splits=5test_size=0.2random_state=426random_seed=51423algo_string=decCFsim_string=Pearson.pk"
	#filled_graph = load_pickled_instance(DAG, pickle_path3)
	#children_set = collect_all_children(filled_graph, "3", t = None)
	#print(children_set)


	snapshot_output_path1 = "../data/snapshots/2019_12_03_18_11_09_t=5_futureSnapshotsDecAggCF"

	#snapshot_df_602 = pd.read_csv(os.path.join(snapshot_output_path1, "602.csv"), index_col = 0)
	#print(snapshot_df_602.head(5))
	#unq_items = snapshot_df_602["itemId"].drop_duplicates()

	#for unq_item in unq_items:
	#	item_snapshot_df = snapshot_df_602[snapshot_df_602["itemId"] == unq_item]
	#	if unq_item == 1580:
	#		print(item_snapshot_df)
	#
	#		#if (aggregated_CFformula(item_snapshot_df) <1.0) | (aggregated_CFformula(item_snapshot_df) > 5.0):
	#		#	print(item_snapshot_df)
	#		#	print(unq_item, aggregated_CFformula(item_snapshot_df))


	return


if __name__ == "__main__":
	#main()
	# predefined dataset paths
	ml_smallest_datapath 							= "../data/ml-latest-small/ratings.csv"		# 610 unique users, 9724 unique items
	test_datapath        							= "../data/test"
	test_matrix_datapath 							= "../data/useritem_matrices/test.csv"
	B_90_cosine_hide_10_t_30_neighmatrix_datapath	= "../data/useritem_matrices/B-90-cosine-hide-10-t=30-neighborhooditemmatrix.csv"
	#################################################
	# select datapath
	datapath 										= ml_smallest_datapath#test_datapath#test_datapath#test_datapath#test_datapath#
	neighborhood_item_matrix_path 					= B_90_cosine_hide_10_t_30_neighmatrix_datapath# test_matrix_datapath


	dataset_string	= os.path.split(datapath)[-1]
	n_splits     	= 5
	random_state 	= 426 # for train/test
	foldNr			= 0

	# load rating data as a pandas.DataFrame in coordinate format with the columns ["userId", "itemId", "rating"]
	rating_df = load_ratings_df(datapath)

	# extract unique peers
	peers = rating_df.loc[:,"userId"].drop_duplicates()
	# make train/test split
	folds = make_train_test_split(rating_df, n_splits = n_splits, random_state = random_state, shuffle = True)#
	train_df, test_df = folds[foldNr]
	# truncate for testing
	#test_df = test_df.iloc[:10,:]

	# make predictions dict neighborhood version
	#predictions_dict = make_predictions_dict_neighborhood_version(train_df, test_df, neighborhood_item_matrix_path, k_max = -1)
	predictions_dict = make_predictions_dict_neighborhood_version_inparallel(train_df, test_df, neighborhood_item_matrix_path, k_max = -1)


	#print(test_df)



	# initialize the result pandas.DataFrame
	result_df = pd.DataFrame()

	# MSE
	accuracy_dict, MSE = get_accuracy(predictions_dict)
	# user-space coverage
	user_space_coverage = get_user_space_coverage(predictions_dict, k_items = 10, threshold = 3.5)

	# get the precisions and recalls in a dictionary @K (for recommended lists of K items)
	precisions_dict, recalls_dict, F1_dict, mean_precision, mean_recall, mean_F1 = get_precision_and_recall(predictions_dict, threshold = 3.5, K =10)

	# get the quotient to the RMSE and the user-space coverage
	if user_space_coverage != 0.0:
		RMSE_cov = math.sqrt(MSE)/user_space_coverage
	else:
		RMSE_cov = np.nan

	result_row = pd.DataFrame([[MSE, user_space_coverage, mean_precision, mean_recall, mean_F1, RMSE_cov]])

	result_df = result_df.append(result_row)


	result_df.columns = ["MSE", "user-space cov.", "Precision@10", "Recall@10", "F1@10", "RMSE/cov"]

	print(result_df)
