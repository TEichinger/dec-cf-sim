
# DESCRIPTION: this file contains basic functions
import sys, os
from pathlib import PurePath
file_dir   = PurePath(os.path.dirname(os.path.realpath(__file__)))
src_dir    = str(file_dir.parents[0]) # ./src
root_dir   = str(file_dir.parents[1]) # ./
sys.path.insert(0, src_dir)
# for debugging

# for production
from mobility_models.graph import DAG
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pickle
import math
import random
import timeit
from datetime import datetime

from collections import defaultdict, deque

from itertools import islice, takewhile, repeat, tee
from sklearn.model_selection import KFold

import re

from cornac.data import Dataset


def make_uniform_T_dict(userIds, lower_T_bound, upper_T_bound, random_seed = None):
	random_gen = random.Random(random_seed)
	# lower and upper bound included
	T_dict = {userId:random_gen.randint(lower_T_bound, upper_T_bound) for userId in userIds}

	return T_dict



def make_uniform_percentile_dict(userIds, lower_percentile_bound, upper_percentile_bound, random_seed = None):
	random_gen = random.Random(random_seed)

	percentile_dict = {userId:random_gen.uniform(lower_percentile_bound, upper_percentile_bound) for userId in userIds}

	return percentile_dict



def convert_df_to_cornac_data(df, sender_as_userId = False):
	""" Convert a pandas dataframe into a dataframe usable by the cornac library.

	NOTE For example DecAggCFv6 or DecShCFv2 add the receiver in the "userId" column as the rating originator.
	This is due to the fact that profile aggregation obfuscates who contributed to the aggregated ratings.
	In this case, the "userId" column should not be used, yet the "sender" column instead.

	NOTE as it is possible that a user received multiple ratings on the same item by the same sender, we also pick the latest received rating per item.

	data (array-like, required) – Data in the form of triplets (user, item, rating)
	"""
	# 1. select sub dataframe with (userId, itemId, rating) triples
	# 2. drop duplicates
	if sender_as_userId:
		# 1.
		sub_df = df.loc[:, ["sender", "itemId", "rating"]] # use with for instance DecAggCFv6
		# 2.
		sub_df = sub_df.drop_duplicates(subset = ["sender", "itemId"], keep='last')

	else:
		# 1.
		sub_df = df.loc[:, ["userId", "itemId", "rating"]] # use with DecCF
		# 2.
		sub_df = sub_df.drop_duplicates(subset = ["userId", "itemId"], keep='last')

	# 3. convert to cornac_data format
	cornac_data = sub_df.to_records(index=False).tolist()
	return cornac_data


def safe_mean(np_array):
	""" Calculate the mean of a numpy.array safely, i.e. whenever the array is empty --> return np.nan"""
	if np_array.size == 0:
		return np.nan
	else:
		return np_array.mean()

def sampleN(random_gen, N, groundtruth_list, must_sample_list = None, drop_duplicates = False):
	""" CAVEAT: This is a duplicate of the class function in mobility.py AssignNMobility meant for arbitrary groundtruth_lists


	Sample N distinct peers from a list of peers, unless the number of peers to sample from is smaller
		than N. In this case, return peers.
	Input:
		- random_gen		: instance of random.Random (for reproducible seeded random samples)
		- N					: <int> number of samples to draw from peers
		- groundtruth_list	: <list> of elements to sample from (can be of any type), can also handle pd.Series with adequate indices
		- must_sample_list	: <list> of elements that necessarily have to be contained in the output (sub)sample, else None
		- drop_duplicates	: <bool> if True, drop all duplicates in peers before sampling, else not.
	Output:
		- random_sample		: <list> of N elements of peers, if the number of elements in peers to sample from is at least as large as N
		- random_gen		: instance of random.Random

	"""
	original_groundtruth_list = groundtruth_list.copy()
	# if drop_duplicates, remove all duplicates from the peers <list> before sampling
	if drop_duplicates:
		groundtruth_list = pd.Series(groundtruth_list).drop_duplicates().tolist()
	# trivial case of drawing N out of less than N items
	if len(groundtruth_list)<N:
		return groundtruth_list, random_gen
	# initialize list containing the random_sample
	if must_sample_list:
		random_sample = must_sample_list[:N]
		N -= len(must_sample_list)
	else:
		random_sample = []

	# for all (remaining) N
	for i in range(N):
		# sample one, and adjust the peers (list) to sample from
		sampled_element, random_gen, groundtruth_list = sample1(random_gen, groundtruth_list)
		random_sample.append(sampled_element)

	return random_sample, random_gen, original_groundtruth_list

def sample1(random_gen, groundtruth_list):
	"""   CAVEAT: This is a duplicate of the class function in mobility.py AssignNMobility
	NOTE Please use sampleN even if you would like to only sample a single peer!

	Sample 1 distinct peer from a list of peers.
	Input:
		- random_gen		: instance of random.Random
		- groundtruth_list	: <list> of elements to draw from, can also handle pd.Series with adequate indices
	Output:
		- sampled_peer		: The sampled single element of (original) peers
		- random_gen		:
		- remainder_peers	: <list> of the remainder of elements (original peers without sampled_peer)

	"""
	random_number = random_gen.random()
	# random_numer in [0.0, 1.0)
	num_elements = len(groundtruth_list)
	random_number *= num_elements
	# random_number in [0.0, num_peers)
	random_index = math.floor(random_number)
	# select the corresponding peer
	sampled_element = groundtruth_list[random_index]
	# remove the sampled peer from the peers
	groundtruth_list.pop(random_index)
	remainder_groundtruth_list = groundtruth_list
	return sampled_element, random_gen, remainder_groundtruth_list



def make_train_test_split_per_user(rating_df, train_frac = 0.8, traintest_seed = 1, username_col_name = "userId"):
	"""
	Split a rating_df (pandas.DataFrame) in the usual coordinate format (columns: "userId", "itemId", "rating") into training and testing set.
	The split procedure samples ratings (a) per user and (b) without replacement. As a consequence, every user has <train_frac> training ratings
	and 1-<train_frac> testing ratings.

	If you are looking for splitting behavior that does not consider users and samples over all user ratings, use the <make_train_test_split> function.

	Input:
		- rating_df: pandas.DataFrame im coordinate format

	Output:
		- folds		: list of duples, where every duple is of the form (train_df, test_df) (both are pandas.Dataframe)
	"""

	train_df = rating_df.groupby(username_col_name).sample(frac=train_frac, random_state = traintest_seed)
	test_df_indices =  set(rating_df.index).difference(set(train_df.index))
	test_df = rating_df.loc[test_df_indices,:]

	return train_df, test_df



def make_train_test_split(rating_df, n_splits = 5, traintest_seed = 1, shuffle = True, foldNr = 0):
	""" Split a rating_df (pandas.DataFrame) in the usual coordinate format (columns: "userId", "itemId", "rating") into training and testing set.
		The split procedure is independent (!) from userId's and samples without replacement from all ratings. As a consequence, it can occur
		that a user has only ratings in the training set.

		If you are looking for splitting behavior that considers users and samples per user, use the <make_train_test_split_per_user> function.

	Input:
		- rating_df: pandas.DataFrame im coordinate format

	Output:
		- folds		: list of duples, where every duple is of the form (train_df, test_df) (both are pandas.Dataframe)
	"""
	kf = KFold(n_splits=n_splits, random_state = traintest_seed, shuffle = shuffle)
	folds = []
	for train_index, test_index in kf.split(rating_df):
		train_df, test_df = rating_df.loc[train_index,:], rating_df.loc[test_index,:]
		folds.append((train_df, test_df))

	train_df, test_df = folds[foldNr]

	return train_df, test_df



def build_execution_graph(peers, mobility_model):
	""" For a set of peers and an initialized mobility model (e.g. AssignNMobility),
	generate the execution graph for the decentralized CF simulator. """
	graph = DAG(graph_dict = None, peers = peers, T = None)
	# generate graph
	mobility_model.generate_graph()
	graph = mobility_model.graph
	# use a mobility_model (for rewiring pruned edges; see <graph.prune_by_t>)
	graph.use_mobility_model(mobility_model)

	return graph




def load_snapshot_df(snapshot_output_path, userId):
	snapshot_df = pd.read_csv(os.path.join(snapshot_output_path, userId+".csv"), index_col = 0, \
	dtype = {'userId': str, 'itemId': str, 'rating': float, 'sim': float, 'agg_sim': float, 'sender': str, 'sim_to_sender': float, 'timestamp': int} )
	return snapshot_df



def load_ratings_df(datapath, drop_timestamp = True, verbose = True, userId_label = "userId", itemId_label = "itemId", rating_label = "rating", timestamp_label = "timestamp"):
	""" Load a ratings file in coordinate format, that is every rating entry is represented by a row and the columns hold userId, itemId, rating, and timestamp.
		For example the ratings.csv file in one of the MovieLens datasets (see for instance http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)

	Input:
		- datapath			: <str> path to the dataset (e.g. *.csv) of ratings in coordinate format to be loaded
		- drop_timestamp	: <bool> True: drop the timestamp column; else do nothing
		- verbose			: <bool> True: print information to command line; else do nothing
		- userId_label		: <str> customizable userId column label
		- itemId_label		: <str> customizable itemId column label
		- rating_label		: <str> customizable rating column label
		- timestamp_label	: <str> customizable timestamp column label
	"""
	dataset = os.path.split(datapath)[1]
	if verbose:
		print("Read in ratings data from {}...".format(datapath))
	# read the data in the form of a table with columns
	# userId 	itemId 	rating 	   timestamp; where timestamp will be dropped if drop_timestamp == True
	#                       np.float64
	rating_df = pd.read_csv(datapath, dtype = {userId_label : str, itemId_label: str, rating_label : float, timestamp_label : int})

	# [OPTIONAL] drop "timestamp" column
	if drop_timestamp & ("timestamp" in rating_df.columns):
		rating_df = rating_df.drop(["timestamp"], axis = 1)

	# cast custom-column labels to standard labels
	rating_df = rating_df.rename(columns= {userId_label: "userId", itemId_label :"itemId", rating_label: "rating", timestamp_label: "timestamp"})

	return rating_df



def df2Snapshot_refs(rating_df, peer_list):
	"""
	Input:
		- rating_df: pandas.DataFrame that is of the format "userId" "itemId" "rating"
		- peer_list: <list>/iterable of <str> identifiers for the "userId"s in rating_df


	Output:
		- snapshot_dict : <dict> with references (pandas.DataFrame.index type) of rows in rating_df.
	"""
	snapshot_dict = dict()
	# intialize initial rating dictionaries
	for peer in peer_list:
		peer_references = rating_df[rating_df["userId"] == peer].index
		snapshot_dict[peer] = peer_references
	return snapshot_dict



def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.
	Taken directly from https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-compute-precision-k-and-recall-k [accessed: Dec6, 2019]


	Input:
		- predictions as the output of a surprise 'algorithm'; list of the format:
				[Prediction(uid='1', iid='3', r_ui=4.0, est=3.1504475058416896, details={'actual_k': 20, 'was_impossible': False}), ..]
		- k			: <int> length of the recommended list
		- threshold : <float> rating (on the rating scale), above which an estimated rating is recommended, an item with true rating above the threshold
						is considered relevant. The opposite for below.
	Output:
		- precisions	: <dict> of recalls (key: userId, value: precision)
		- recalls		: <dict> of recalls (key: userId, value: recall)
	'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

def make_train_and_test_files(rating_df, n_splits = 5, traintest_seed = 0, dataset_name = "", output_dir = "./"):
	""" For a given rating_df, split it with the KFolds class of sklearn into
	a training and a testing data set. Save them as .csv files. """
	# calculate n_splits 'randomized/shuffled' folds (according to the traintest_seed
	kf = KFold(n_splits=n_splits, random_state = traintest_seed, shuffle = True)
	folds = []
	for train_index, test_index in kf.split(rating_df):
		train_df, test_df = rating_df.loc[train_index,:], rating_df.loc[test_index,:]
		folds.append((train_df, test_df))

	# write files
	for i ,fold in enumerate(folds,1):
		fold_name = dataset_name + "_fold={}_nsplits={}_randomstate={}".format(i,n_splits, traintest_seed)
		print("Save {}-th/{} split in {}.".format(i, n_splits, fold_name))


		# unpack fold
		train_df, test_df = fold
		# write train and test df to *.csv files
		train_df.to_csv(os.path.join(output_dir, fold_name) + "_train.csv", index = False)
		test_df.to_csv(os.path.join(output_dir, fold_name) + "_test.csv",   index = False)
	return

def make_sim_dict_pickle_string(dataset_string, train_test_seed, mobility_seed, hide_seed, hide_p, T, sim_string):
	""" Create a filename for a pickle file of a sim_dict for future quick loading such that we do not have to reload the sim_dict every time.
	The following information are necessary:

	dataset, train-test_seed, mobility_seed, hide_seed, hide_p, T
	"""
	if hide_seed is None:
		hide_seed = "none"
	if hide_p is None:
		hide_p = "none"


	sim_dict_pickle_string = "sim_dict_"
	if dataset_string is not None:
		sim_dict_pickle_string += "datastring={}".format(dataset_string)
	if train_test_seed is not None:
		sim_dict_pickle_string += "_ttseed={}".format(str(train_test_seed))
	if mobility_seed is not None:
		sim_dict_pickle_string += "_mobseed={}".format(str(mobility_seed))
	if hide_seed is not None:
		sim_dict_pickle_string += "_hideseed={}".format(str(hide_seed))
	if hide_seed is not None:
		sim_dict_pickle_string += "_hidep={}".format(str(hide_p))
	if T is not None:
		sim_dict_pickle_string += "_T={}".format(str(T))
	if sim_string is not None:
		sim_dict_pickle_string += "_simstring={}".format(sim_string)
	sim_dict_pickle_string += ".pk"

	return sim_dict_pickle_string


def make_initial_snapshot_string(dataset_string, traintest_seed, ttsplit_per_user = False):
	""" Make the initial snapshot string that will be used to persist the per user indices in the train_df.
		For instance in a train_df such as

		userId, itemId, rating
		1,2,5
		2,2,3
		1,3,4

		The indices for user with userId == 1 are [0,2].
	"""
	initial_snapshot_string = ""

	initial_snapshot_string += "_datasetstring={}".format(dataset_string)
	if ttsplit_per_user:
		initial_snapshot_string += "_traintestseed(p.u.)={}".format(traintest_seed)
	else:
		initial_snapshot_string += "_traintestseed={}".format(traintest_seed)

	return initial_snapshot_string


def unmake_snapshot_info_string(snapshot_info_string):
	""" Unmake a snapshot info string created by the <make_snapshot_info_string> function and return its input parameters.

		Input:
			- snapshot_info_string: <str> a string produced by the <make_snapshot_info_string> function.

			Example:

			snapshot_info_string = "2021_11_06_11_38_50_t=5_topN=3_simstring=cosine_datasetstring=movies.csv_graphseed=51423
									_nsplits=5_traintestseed=426_N=3_mobString=AssignNMobility_perc=0.5_dynMinSimToSender=True_idContacts_DecCF"

		Output:
			- topN
			- sim_string
			- dataset_string
			- graph_seed
			- n_splits
			- traintest_seed
			- N
			- mobility_string
			- percentile
			- timestamp_delta
			- min_sim_to_sender_dynamic
			- anonymous_contacts

			Example:

	"""
	result_dict = dict()

	# t; NOTE that t is only added in <DecAlgoTemplate.save_future_snapshots>
	t = extract_variable_from_info_string('t=[0-9]+', snapshot_info_string)
	if t is not None:
		result_dict["t"] = int(t)
	else:
		result_dict["t"] = "none"

	# topN
	topN = extract_variable_from_info_string('topN=[0-9]+', snapshot_info_string)
	if topN is not None:
		result_dict["topN"] = int(topN)
	else:
		result_dict["topN"] = "none"

	#sim_string
	sim_string = extract_variable_from_info_string('simstring=\w+', snapshot_info_string)
	result_dict["sim_string"] = sim_string

	#dataset_string
	dataset_string = extract_variable_from_info_string('datasetstring=[\w]+', snapshot_info_string)
	result_dict["dataset_string"] = dataset_string

	#graph_seed
	graph_seed = extract_variable_from_info_string('graphseed=[0-9]+', snapshot_info_string)
	if graph_seed is not None:
		graph_seed = int(graph_seed)
	result_dict["graph_seed"] = graph_seed

	# n_splits
	n_splits = extract_variable_from_info_string('nsplits=[0-9]+', snapshot_info_string)
	if n_splits is not None:
		n_splits = int(n_splits)
	result_dict["n_splits"] = n_splits

	# traintest_seed
	traintest_seed = extract_variable_from_info_string('traintestseed=[0-9]+', snapshot_info_string) #TODO: add per user mod "(p.u.)"
	if traintest_seed is not None:
		traintest_seed = int(traintest_seed)
	result_dict["traintest_seed"] = traintest_seed

	# N
	N = extract_variable_from_info_string('N=[0-9]+', snapshot_info_string)
	if N is not None:
		N = int(N)
	result_dict["N"] = N

	# mobility_string
	mobility_string = extract_variable_from_info_string('mobString=[\w]+', snapshot_info_string)
	result_dict["mobility_string"] = mobility_string

	# percentile
	percentile = extract_variable_from_info_string('perc=[0-9]+.[0-9]+', snapshot_info_string)
	if percentile is not None:
		percentile = float(percentile)
	result_dict["percentile"] = percentile

	# timestamp_delta
	timestamp_delta = extract_variable_from_info_string('timestampDelta=[0-9]+', snapshot_info_string)
	if timestamp_delta is not None:
		timestamp_delta = int(timestamp_delta)
	result_dict["timestamp_delta"] = timestamp_delta


	# min_sim_to_sender_dynamic
	min_sim_to_sender_dynamic = extract_variable_from_info_string('dynMinSimToSender=\w+', snapshot_info_string)
	if min_sim_to_sender_dynamic is not None:
		min_sim_to_sender_dynamic = bool(min_sim_to_sender_dynamic)
	result_dict["min_sim_to_sender_dynamic"] = min_sim_to_sender_dynamic

	# anonymous_contacts
	anonContacts = extract_variable_from_info_string('anonContacts', snapshot_info_string, no_value = True)
	if anonContacts is None:
		result_dict["anonymous_contacts"] = False

	return result_dict

def extract_variable_from_info_string(pattern, info_string, no_value = False):
	"""
		Submit a regular expression pattern to extract the value of a variable from an info_string.

	"""
	var_val_strings = info_string.split("_") # similar to ['t=5', 'topN=3', 'simstring=cosine',... ]

	# example var_val_string: 't=5'
	matching_var_val_strings = [var_val_string for var_val_string in var_val_strings if re.match(pattern, var_val_string) is not None]

	if len(matching_var_val_strings)>0:
		# if the string only has a variable e.g. "[...]_anonContacts_[...]"
		if no_value:
			val_string = "True"
		# if the string has a value declaration e.g. "[...]_graphseed=1234_[...]"
		else:
			# extract value of the first (!) var_val_string: e.g. 't=5' -> val: '5'
			# the .join is needed to catch for instance dataset strings such as 'ml-25m:n=500:rseed=1234.csv'
			val_string = '='.join(matching_var_val_strings[0].split("=")[1:])
		return val_string
	else:
		#print("ERROR: could not find pattern:{}".format(pattern))
		return None


def make_snapshot_info_string(topN, sim_string, dataset_string, graph_seed, n_splits, traintest_seed, N, mobility_string,\
								percentile, hide_p = None, p = None, timestamp_delta = None, min_sim_to_sender_dynamic = None,\
								anonymous_contacts = False, ttsplit_per_user = False, mean_centering = False, parameter_control_string = None):
	""" Make a snapshot string to pass to a simulation class such as DecentralizedAggregatingCF.
	NOTE the string inputs shall not contain '/', '..' or any other path indicators for else the snapshots will not be able to be saved
	Input:
		- topN
		- sim_string
		- dataset_string
		- graph_seed
		- n_splits
		- traintest_seed
		- N
		- mobility_string
		- percentile
		- timestamp_delta
		- min_sim_to_sender_dynamic
		- anonymous_contacts
		- parameter_control_string

		CAVEAT: it seems that the max. path length is limited, therefore, the names should be as short as possible.
				OSError: [Errno 36] File name too long
	"""
	snapshot_info_string = ""
	if topN is not None:
		snapshot_info_string += "_topN={}".format(topN)
	if sim_string is not None:
		snapshot_info_string += "_simstring={}".format(sim_string)
	if dataset_string is not None:
		snapshot_info_string += "_datasetstring={}".format(dataset_string)
	if graph_seed is not None:
		snapshot_info_string += "_graphseed={}".format(str(graph_seed))
	if n_splits is not None:
		snapshot_info_string += "_nsplits={}".format(str(n_splits))
	if traintest_seed is not None:
		if ttsplit_per_user:
			snapshot_info_string += "_traintestseed(p.u.)={}".format(str(traintest_seed))
		else:
			snapshot_info_string += "_traintestseed={}".format(str(traintest_seed))
	if N is not None:
		snapshot_info_string += "_N={}".format(str(N))
	if mobility_string is not None:
		snapshot_info_string += "_mobString={}".format(str(mobility_string))
	if percentile is not None:
		snapshot_info_string += "_perc={}".format(str(percentile))
	#if hide_p is not None:
	#	snapshot_info_string += "_hidep={}".format(str(hide_p))
	#if p is not None:
	#	snapshot_info_string += "_p={}".format(str(p))
	if timestamp_delta is not None:
		snapshot_info_string += "_timestampDelta={}".format(str(timestamp_delta))
	if min_sim_to_sender_dynamic is not None:
		snapshot_info_string += "_dynMinSimToSender={}".format(str(min_sim_to_sender_dynamic))
	if anonymous_contacts == False:
		snapshot_info_string += "_idContacts"
	else:
		snapshot_info_string += "_anonContacts"
	if mean_centering == True:
		snapshot_info_string += "_(m.c.)"
	if parameter_control_string is not None:
		snapshot_info_string += "_param={}".format(parameter_control_string)


	return snapshot_info_string


def split_every_n(n, iterable):
    """
    Slice an iterable into chunks of n elements
    :type n: int
    :type iterable: Iterable
    :rtype: Iterator
    """
    iterator = iter(iterable)
    return takewhile(bool, (list(islice(iterator, n)) for _ in repeat(None)))


# Using the generator pattern (an iterable)
class RandomBinaryIterator(object):
	""" Generator that iterates a series of True, False <boolean> values, either with a designated seed
	or truly random.
		- random_seed: seed to use for reproducibility; no reproducibility when used with None as seed
		- p          : probability that the next element in the iterator is True (independent of the previous outcomes).
	"""
	def __init__(self, p, random_seed=None):
		self.__p           = p
		self.__random_seed = random_seed
		self.__random_gen  = random.Random(self.__random_seed)

	def __iter__(self):
		return self

	# Python 3 compatibility
	def __next__(self):
		return self.next()

	def next(self):
		return self.bernoulli()[1]

	def bernoulli(self):
		""" Conduct a Bernoulli experiment and reports its result as boolean (True/False)
		Probability   p of success: True
		Probabiltiy 1-p of failure: False """
		# roll number
		real_number = self.__random_gen.random()
		if real_number < self.__p:
			return self.__random_gen, True
		else:
			return self.__random_gen, False

# Using the generator pattern (an iterable)
class UpperTriangleGeneratorY(object):
	""" Generator that iterates over the upper triangle of a matrix when given the matrix indices. More specifically,
		the elements of a list are paired such that pairs of elements (duples) are unique modulo symmetry, for instance (1,2) = (2,1).

	Input:
		- matrix_indices:	<list>, for which pairs of elements (duples) are to be created that are unique modulo symmetry, for instance (1,2) = (2,1).

	Output:
		- generator that iterates over the upper triangle of the following matrix:

		matrix_indices = ['a', 'b', 'c', 'd']


			'a' 'b' 'c' 'd'
		'a'	x	1	2	3
		'b'	x	x	4	5
		'c'	x	x	x	6
		'd'	x	x	x	x

		-> [('a','b'), ('a', 'c'), ('a','d'), ('b','c'), ('b','d'), ('c','d'), .. None]

	"""
	def __init__(self, matrix_indices):
		self.matrix_indices  = matrix_indices
		self.subgen          = deque(matrix_indices[1:])
		self.subsubgen       = deque(self.matrix_indices[1:])
		self.sub_index       = matrix_indices[0]
		self.subsub_index    = None

	def __iter__(self):
		return self

	# Python 3 compatibility
	def __next__(self):
		return self.next()

	def next(self):
		try:
			self.subsub_index = self.subsubgen.popleft()
		except:
			try:
				self.sub_index = self.subgen.popleft()
				self.subsubgen = deque(self.subgen.copy())
				try:
					self.subsub_index = self.subsubgen.popleft()
				except:
					raise StopIteration()
			except:
				raise StopIteration()
		return self.subsub_index



# Using the generator pattern (an iterable)
class UpperTriangleGeneratorX(object):
	""" Generator that iterates over the upper triangle of a matrix when given the matrix indices. More specifically,
		the elements of a list are paired such that pairs of elements (duples) are unique modulo symmetry, for instance (1,2) = (2,1).

	Input:
		- matrix_indices:	<list>, for which pairs of elements (duples) are to be created that are unique modulo symmetry, for instance (1,2) = (2,1).

	Output:
		- generator that iterates over the upper triangle of the following matrix:

		matrix_indices = ['a', 'b', 'c', 'd']


			'a' 'b' 'c' 'd'
		'a'	x	1	2	3
		'b'	x	x	4	5
		'c'	x	x	x	6
		'd'	x	x	x	x

		-> [('a','b'), ('a', 'c'), ('a','d'), ('b','c'), ('b','d'), ('c','d'), .. None]


	"""
	def __init__(self, matrix_indices):
		self.matrix_indices  = matrix_indices
		self.subgen          = deque(matrix_indices[1:])
		self.subsubgen       = deque(self.matrix_indices[1:])
		self.sub_index       = matrix_indices[0]
		self.subsub_index    = None

	def __iter__(self):
		return self

	# Python 3 compatibility
	def __next__(self):
		return self.next()

	def next(self):
		try:
			self.subsub_index = self.subsubgen.popleft()
		except:
			try:
				self.sub_index = self.subgen.popleft()
				self.subsubgen = deque(self.subgen.copy())
				try:
					self.subsub_index = self.subsubgen.popleft()
				except:
					raise StopIteration()
			except:
				raise StopIteration()
		return self.sub_index


# Using the generator pattern (an iterable)
class UpperTriangleGenerator(object):
	""" Generator that iterates over the upper triangle of a matrix when given the matrix indices. More specifically,
		the elements of a list are paired such that pairs of elements (duples) are unique modulo symmetry, for instance (1,2) = (2,1).

	Input:
		- matrix_indices:	<list>, for which pairs of elements (duples) are to be created that are unique modulo symmetry, for instance (1,2) = (2,1).

	Output:
		- generator that iterates over the upper triangle of the following matrix:

		matrix_indices = ['a', 'b', 'c', 'd']


			'a' 'b' 'c' 'd'
		'a'	x	1	2	3
		'b'	x	x	4	5
		'c'	x	x	x	6
		'd'	x	x	x	x

		-> [('a','b'), ('a', 'c'), ('a','d'), ('b','c'), ('b','d'), ('c','d'), .. None]


	"""
	def __init__(self, matrix_indices):
		self.matrix_indices  = matrix_indices
		self.subgen          = deque(matrix_indices[1:])
		self.subsubgen       = deque(self.matrix_indices[1:])
		self.sub_index       = matrix_indices[0]
		self.subsub_index    = None

	def __iter__(self):
		return self

	# Python 3 compatibility
	def __next__(self):
		return self.next()

	def next(self):
		try:
			self.subsub_index = self.subsubgen.popleft()
		except:
			try:
				self.sub_index = self.subgen.popleft()
				self.subsubgen = deque(self.subgen.copy())
				try:
					self.subsub_index = self.subsubgen.popleft()
				except:
					raise StopIteration()
			except:
				raise StopIteration()
		return (self.sub_index, self.subsub_index)



def safe_dict_norm(rating_dictionary, aggregated = False):
	""" Calculate the Euclidean norm of a rating dictionary. """
	if aggregated:
		values = list([el[0] for el in rating_dictionary.values()])
	else:
		values = list(rating_dictionary.values())
	return safe_norm(values)


def safe_normalize_dict(rating_dictionary):
	""" Safely normalize (Euclidean) of a rating dictionary. Note that the values of the dictionary cannot be None"""
	result_dict = dict()

	keys   = rating_dictionary.keys()
	values = list(rating_dictionary.values())

	normalized_ratings = safe_normalize_vec(values)

	for key, normalized_rating in zip(keys, normalized_ratings):
		result_dict[key] = normalized_rating

	return result_dict


def safe_norm(vector):
	""" Calculate the Euclidean norm. If the norm is 0, return 1 instead. (to prevent division by zero)"""
	norm = np.linalg.norm(vector)
	if norm == 0.0:
		return 1.0
	return norm

def safe_normalize_vec(vector):
	""" Normalize a list (vector) according to the Euclidean norm. Return the zero vector when the vector norm is zero.
	The output is a Python float.

	"""
	norm = np.linalg.norm(vector)
	if norm == 0.0:
		return [0.0 for _ in vector]
	else:
		return [el/norm for el in vector]

# normalize a list (vector) of positives such that they sum to 1
# return the zero vector when the vector norm is zero
def safe_normalize_histo(vector):
	denominator = sum(vector)
	if denominator == 0.0:
		return [0.0 for _ in vector]
	else:
		return [el/denominator for el in vector]


# define safe division, that is in case of division by zero, the result is zero
def safe_division(numerator, denominator):
	if denominator == 0.0:
		return 0.0
	else:
		return numerator/denominator

def save_pickle(obj):
	""" Save a class instance <obj> as a duple of the class object and its attribute dictionary.
	Used for pickling. Recall that pickle does not pickle code such that the class-defining code has_key
	to be available at unpickling time. """
	return (obj.__class__, obj.__dict__)

def load_class_attributes(cls, attributes):
	""" Take a class object and apply a attribute dictionary (that had been pickled) and apply it to the
	corresponding class in order to regain the original class instance to be saved. """
	obj = cls.__new__(cls)
	obj.__dict__.update(attributes)
	return obj

def pickle_instance(class_instance, pickle_path):
	""" Pickle a class instance's attributes at a designated pickle_path. """
	with open(pickle_path, mode = "wb") as f:
		pickle.dump(class_instance.__dict__, f)
	return

def load_pickled_instance(obj_class, pickle_path):
	""" Load a previously pickled attribute dictionary for the <obj_class> class. """
	with open(pickle_path, mode = "rb") as f:
		attribute_dict = pickle.load(f)
		class_instance = load_class_attributes(obj_class, attribute_dict)
		return class_instance



def cosine_similarity(x1,x2):
	""" Safely calculate the cosine similarity between two vectors of type <list>
	 cos_distance(x1,x2) = 1 - cos(x1,x2)    (in [0,1])
	 identical non-zero vectors --> 1
	 orthogonal vectors         --> 0
	 opposing vectors			 --> -1"""
	norm_x1 = np.linalg.norm(x1)
	norm_x2 = np.linalg.norm(x2)
	if not (norm_x1 == 0 and norm_x2 == 0):
		dot_product = np.dot(x1,x2)
		cosine_distance = dot_product / (norm_x1 * norm_x2)
	else:
		cosine_distance = 0.0 # None
	return cosine_distance

def coord_cosine(coord_df1, coord_df2, adjust_mean = False, aggregated = False):
	""" Calculate the cosine similarity between two users represented by their rating dictionaries.
	The coord_df's should contain the columns "itemId" and "rating" in order for this function to work.


	Input:
		coord_df1: pandas DataFrame
		coord_df2: pandas DataFrame

		format
			"userId"   "itemId"    "rating"
		0    ...
		1
		2

	CAVEAT: Problems in (*) arise, when the userId is identical, whereas the comparing entities are not the same.
			In other words, make sure that similarity comparison only happens on initial_dfs (as subdataframes of train_df).
	"""
	# trivial case of empty vectors
	if (coord_df1.size == 0) | (coord_df2. size == 0):
		return 0.0
	# extract userId from the first userId entry (necessarily the coord_df1/2 have to be for a single user)
	userId1 = coord_df1.loc[coord_df1.index[0], "userId"]
	userId2 = coord_df2.loc[coord_df2.index[0], "userId"]

	# (*)
	if userId1 == userId2:
		return 1.0

	# if userId1 and userId2 are identical, this crashes!
	# convert format
	d1 = coord_df1.pivot(index='itemId', columns='userId', values='rating')
	d2 = coord_df2.pivot(index='itemId', columns='userId', values='rating')

	# perform inner join
	joined = d1.join(d2, how = "outer").fillna(0.0)# suffixed have to be specified when userId1 and userId2 are identical, which is not allowed!, lsuffix = "l", rsuffix = "r")
	res_d1 = joined.loc[:,userId1]
	res_d2 = joined.loc[:,userId2]
	if adjust_mean:
		res_d1 = res_d1 - res_d1.mean()
		res_d2 = res_d2 - res_d2.mean()

	# calculate cosine similarity # .values selects the underlying numpy.array
	cosine_sim =  cosine_similarity(res_d1, res_d2)
	return cosine_sim



def get_current_timestamp():
	""" Return the current time as a timestamp in the format "%Y_%m_%d_%H_%M_%S" """
	return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")



def coord_Pearson(coord_df1, coord_df2, adjust_mean = False, aggregated = False):
	""" Calculate the Pearson similarity between two users represented by their rating dictionaries.
	The coord_df's should contain the columns "itemId" and "rating" in order for this function to work.

	Input:
		coord_df1: pandas DataFrame
		coord_df2: pandas DataFrame

		format
			"userId"   "itemId"    "rating"
		0    ...
		1
		2

	CONVENTION: similarity is 1.0, if the pearson correlation is nan (see example below); in particular if res_d1
				(and thus also res_d2) only contains one rating
	CONVENTION: when userId1 == userId2, then the similarity defaults to 1. This scenario is that parent and child_child are identical
				This is possible when e.g. two peers meet each other twice! (cf. (*))
	"""
	# trivial case of empty vectors
	if (coord_df1.size == 0) | (coord_df2. size == 0):
		return 0.0
	# extract userId from the first userId entry (necessarily the coord_df1/2 have to be for a single user)
	userId1 = coord_df1.loc[coord_df1.index[0], "userId"]
	userId2 = coord_df2.loc[coord_df2.index[0], "userId"]

	# if the userId1 and userId2 are identical, then the similarity is trivially 1.0
	if userId1 == userId2:
		return 1.0

	# if userId1 and userId2 are identical, this crashes!
	# convert format
	d1 = coord_df1.pivot(index='itemId', columns='userId', values='rating')
	d2 = coord_df2.pivot(index='itemId', columns='userId', values='rating')
	# perform inner join
	joined = d1.join(d2, how = "inner")# suffixed have to be specified when userId1 and userId2 are identical, which is not allowed!, lsuffix = "l", rsuffix = "r")

	res_d1 = joined.loc[:,userId1]
	res_d2 = joined.loc[:,userId2]

	# no commonly rated items
	if (res_d1.size == 0) | (res_d2.size==0): #[the second part necessarily is True when the former is]
		return 0.0
	else: # more than 1 commonly rated item
		pearson_correlation = custom_pearsonr(res_d1, res_d2)
		# special case in which the pearson correlation cannot be estimated since the standard deviations are zero
		# e.g. if
		# itemId  rating
		# 2716    3.0
		# 3114    5.0
		# itemId  rating
		# 2716    4.0
		# 3114    4.0
		if np.isnan(pearson_correlation):
			return 1.0
		else:
			return pearson_correlation

def custom_pearsonr(x, y):
	""" Customized code of scipy.stats' pearsonr source code.
	Here x, y must be vectors (pandas Series or a numpy array (implementing .mean())
	Moreover, the size of both vectors must be equal.

	Input:
		- x,y : numpy.array/pandas.Series of the same length
	Output:
		- r   : Pearson correlation [float between -1 and 1]


	formula:
		Pearsoncorrelation =    E[(X-E[X]) * (Y-E[Y])]   =                1/n * <xm,ym>                  =               <xm,ym>
								_______________________    ___________________________________________         _______________________

								std(x)     * std(y)         sqrt(1/n * <xm,xm>) *  sqrt(1/n * <ym,ym>)         sqrt(<xm,xm> * <ym,ym>)

	NOTE that the sums are all over the SHARED PORTION (!) of items only!
	"""
	n = len(x)

	# only consider those elements that have non-zero ratings in both x and y
	# lambda function that maps all zero items to zero, and and all non-zero items to 1
	to_one = lambda el: 0 if el == 0.0 else 1

	# e.g. x = np.ndarray([1,5,0,0,4,0]) --> x_indicator = np.ndarray([1,1,0,0,1,0]), hence multiplication yields non-zero elements only on commonly rated items
	indicator_function		= np.vectorize(to_one)
	x_indicator				= indicator_function(x)
	y_indicator				= indicator_function(y)

	x = x*y_indicator
	y = y*x_indicator

	num_shared_items = np.count_nonzero(x) # == np.countnonzero(y)

	if num_shared_items == 0:
		return 0.0

	# mean centering of x and y; only take the mean over the shared items!
	mx = x.sum()/num_shared_items
	my = y.sum()/num_shared_items

	mean_center_x = lambda el: el-mx if el != 0 else 0
	mean_center_y = lambda el: el-my if el != 0 else 0
	mean_center_x_function = np.vectorize(mean_center_x)
	mean_center_y_function = np.vectorize(mean_center_y)

	# mean_center (only non_zero elements)
	xm = mean_center_x_function(x)
	ym = mean_center_y_function(y)
	# denominator r_den np.float64
	# we omit dividing both the numerator and the denominator by n
	r_den = math.sqrt(np.dot(xm,xm) * np.dot(ym,ym))

	# when the denominator is 0.0 and the quotient (r) would default to a NaN, return 0.0 instead
	if r_den == 0.0: # r_den == 0.0 is equivalent of x or y having zero variance
		return 0.0 # recently 1.0
	# numerator
	r_num = np.dot(xm, ym)
	# Covariance between x and y divided by the product of their respective standard deviations
	r = (r_num / r_den)
	return r

def dict_Pearson(rating_dict1, rating_dict2, adjust_mean = False, aggregated = False):
	""" Calculate the Pearson similarity between two users represented by their rating dictionaries.

	NOTE Pearson correlation and adjusted cosine similarity are very similar. However, they are per se not idential.
		 We can think of Pearson correlation as mean_adjusted cosine, where the mean is not calculated overt
		 the entire set of rated items of users u1 and u2, yet only (!) on the common items.
		 In pathological cases, i.e. when the intersection set is a single item, using mean_adjusted cosine seems to be
		 more meaningful since else it is degenerate (=0, even though there is a rating match).

	Format:
		- rating_dict : {itemId: rating}
		- adjust_mean : False [dummy kwarg, in order to be compatible (function-signature) to <dict_cosine>
	"""
	# shared item keys for items both users u1 and u2 represented by rating_dict1, and rating_dict2
	shared_keys = set(rating_dict1.keys()).intersection(set(rating_dict2.keys()))
	# if the dictionaries do not overlap, the similarity defaults to 0.0
	if shared_keys == set():
		norm1 = 1.0
		norm2 = 1.0
		dot_prod = 0.0
	else:
		if aggregated: # e.g. rating_dict1[key] = [5.0, None, '540']
			vector1 = np.array([rating_dict1[key][0] for key in shared_keys])
			vector2 = np.array([rating_dict2[key][0] for key in shared_keys])
		else:          # e.g. rating_dict1[key] = 5.0
			vector1 = np.array([rating_dict1[key] for key in shared_keys])
			vector2 = np.array([rating_dict2[key] for key in shared_keys])
		vector1_adjusted = vector1 - vector1.mean()
		vector2_adjusted = vector2 - vector2.mean()
		norm1 = safe_norm(vector1)
		norm2 = safe_norm(vector2)
		dot_prod = np.dot(vector1, vector2)
	return dot_prod/(norm1 * norm2)



def drop_nan(df):
	""" Drop rows that contain np.nan's in a numpy array. """
	return df[~np.isnan(df).any(axis=1)]



def dict_cosine(rating_dict1, rating_dict2, adjust_mean = False, aggregated = False):
	""" Calculate the cosine similarity between two rating dictionaries.
	Format dict:  {"itemId1": rating1, "itemId2": rating2, ...}

	The calculation in python terms is faster than converting to a numpy array first and
	using numpy's dot product.

	adjust_mean : True --> Mean adjustment is by subtracting the mean over the entire vectors
	instead of only the ratings on commonly rated items.
	"""
	# shared item keys for items both users u1 and u2 represented by rating_dict1, and rating_dict2
	shared_keys = set(rating_dict1.keys()).intersection(set(rating_dict2.keys()))

	if aggregated: # rating_dict1[key] = [5.0, None, '540']
		vector1 = np.array([[rating_dict1[key][0]] for key in shared_keys])
		vector2 = np.array([[rating_dict2[key][0]] for key in shared_keys])
	else: # rating_dict1[key] = 5.0
		vector1 = np.array([rating_dict1[key] for key in shared_keys])
		vector2 = np.array([rating_dict2[key] for key in shared_keys])

	# take mean over all rated items
	mean1 = np.array(list([el[0] for el in rating_dict1.values()])).mean()
	mean2 = np.array(list([el[0] for el in rating_dict2.values()])).mean()
	if adjust_mean:
		# if the dictionaries do not overlap, the similarity defaults to 0.0
		if shared_keys == set():
			norm1 = 1.0
			norm2 = 1.0
			dot_prod = 0.0
		else:
			# adjust mean
			vector1_adjusted = vector1 - mean1
			vector2_adjusted = vector2 - mean2
			norm1 = safe_norm(vector1)
			norm2 = safe_norm(vector2)
			dot_prod = np.dot(vector1, vector2)
	else:
		norm1 = safe_dict_norm(rating_dict1, aggregated = aggregated)
		norm2 = safe_dict_norm(rating_dict2, aggregated = aggregated)

		dot_prod = 0
		for key in shared_keys:
			if aggregated:
				dot_prod += rating_dict1[key][0] * rating_dict2[key][0]
			else:
				dot_prod += rating_dict1[key]*rating_dict2[key]

	return dot_prod/(norm1 * norm2)


def make_pickle_name(dataset,p,T,N,fold_nr,n_splits,test_size,traintest_seed,random_seed, algo_string, sim_string):
	""" Create a pickle name for a given set of parameters (for the creation of the random graph). """

	pickle_name = "dataset={}".format(dataset)
	pickle_name += "p={}".format(str(p))
	pickle_name += "T={}".format(str(T))
	pickle_name += "N={}".format(str(N))
	pickle_name += "fold_nr={}".format(str(fold_nr))
	pickle_name += "n_splits={}".format(str(n_splits))
	pickle_name += "test_size={}".format(str(test_size))
	if traintest_seed != None:
		pickle_name += "traintest_seed={}".format(str(traintest_seed))
	if random_seed != None:
		pickle_name += "random_seed={}".format(str(random_seed))
	pickle_name += "algo_string={}".format(algo_string)
	pickle_name += "sim_string={}".format(sim_string)
	pickle_name += ".pk"
	return pickle_name




def make_vector():
	r_gen = random.Random(153)

	vector = [r_gen.random() for _ in range(10000)]
	return vector





def main():
	snapshot_info_string = "2021_11_06_11_38_50_t=5_topN=3_simstring=cosine_datasetstring=movies.csv_graphseed=51423_nsplits=5_traintestseed=426_N=3_mobString=AssignNMobility_perc=0.5_dynMinSimToSender=True_idContacts_DecCF"

	res = unmake_snapshot_info_string(snapshot_info_string)
	print(res)


	"""
	groundtruth_list = list(range(30))
	N = 5
	seed = 135
	random_gen = random.Random(seed)
	for i in range(5):
		res, random_gen, groundtruth_list  = sampleN(random_gen, N, groundtruth_list, must_sample_list = None, drop_duplicates = False)
		print(res)

	#print(custom_pearsonr(np.array([1.,5.,4.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]), np.array([2.,3.,5.,3.,5.,0.,0.,0.,0.,0.,0.,0.,0.,0.])))
	#print(custom_pearsonr(np.array([1,5,4]), np.array([2,3,5])))
	#print(custom_pearsonr(np.array([3,1]), np.array([5,1])))
	"""

	"""

	# MAKE TRAIN AND TEST DATASETS
	#
	# LOAD MOVIELENS data (100K)
	################################
	datapath = "../../data/ml-latest-small/ratings.csv"
	dataset = os.path.split(datapath)[1]
	print("Read in data : {}...".format(datapath))
	# read the data in the form of a table with columns
	# user id 	item id 	rating 	   timestamp; where timestamp will not be used
	#                       np.float64
	rating_df = pd.read_csv(datapath)
	# drop timestamp
	rating_df = rating_df.drop(["timestamp"], axis = 1)
	# cast the "userId" to type str
	rating_df["userId"] = rating_df["userId"].astype(str)
	# cast "movieId" to "itemId"
	rating_df = rating_df.rename(columns= {"movieId":"itemID", "itemId" : "itemID"})
	# make train and test files
	make_train_and_test_files(rating_df, n_splits = 5, traintest_seed = 426, dataset_name = "ml-smallest", output_dir = "../../data/ml-latest-small")

	"""

	"""
	coord_d1 = pd.DataFrame([["a", "1", 4.0],["e","1",3.0] , ["d", "1", 4.0]], columns = ["itemId", "userId","rating"])
	coord_d2 = pd.DataFrame([["a", "2",4.0] ,["e", "2", 3.0], ["c", "2", 5.0]], columns = ["itemId", "userId","rating"])
	print(coord_Pearson(coord_d1, coord_d2))#, adjust_mean = False, aggregated = False)
	"""


	"""
	# Create 1000 pandas dataframes of the same column names
	global np_frames
	np_frames = [np.random.randint(5, size = (1,7)) for _ in range(1000)]

	stmt = "np.concatenate(np_frames, axis = 0)"
	print("Concatenating 1000 numpy rows")
	print(timeit.timeit(stmt=stmt, number = 1000, globals = globals()))

	global pd_frames
	pd_frames = [pd.DataFrame(np.random.randint(5, size=(1,7)), columns = ["a", "b", "c", "d", "e", "f", "g"]) for _ in range(1000)]

	stmt = "pd.concat(pd_frames, axis = 0, sort = False, ignore_index = True)"
	print("Concatenating 1000 pandas rows")
	print(timeit.timeit(stmt=stmt, number = 1000, globals = globals()))
	"""
	"""
	# check whether slicing or calling via an index set is faster

	datapath = "../../data/ml-20m/ratings.csv"
	dataset = os.path.split(datapath)[1]
	print("Read in data : {}...".format(datapath))
	# read the data in the form of a table with columns
	# user id 	item id 	rating 	   timestamp; where timestamp will not be used
	#                       np.float64
	global rating_df

	rating_df = pd.read_csv(datapath, nrows = 200000)
	# drop timestamp
	rating_df = rating_df.drop(["timestamp"], axis = 1)
	# cast the "userId" to type str
	rating_df["userId"] = rating_df["userId"].astype(str)
	# cast "movieId" to "itemId"
	rating_df = rating_df.rename(columns= {"movieId":"itemId"})

	# index set
	#print(rating_df.head(10))
	global ind_set
	ind_set = rating_df[rating_df["userId"] == "1054"].index

	stmt = "rating_df.loc[ind_set,:]"
	print(timeit.timeit(stmt = stmt, number = 1000, globals = globals()))

	stmt = "rating_df[rating_df['userId'] == '1054']"

	print(timeit.timeit(stmt = stmt, number = 1000, globals = globals()))
	"""

	return



if __name__ == "__main__":
	main()
