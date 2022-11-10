# Decentralized Collaborative Filtering Simulation (dec-cf-sim)

*This is not a comprehensive Collaborative Filtering library*

*Please refer to the wonderful [Surprise](http://surpriselib.com/) or [Cornac](https://github.com/PreferredAI/cornac) libraries, if you are looking for CF models.*

This Python3-library simulates data dissemination in decentralized collaborative filtering networks, where data is disseminated via pairwise data exchanges between users.
It allows to simulate distinct network topologies and study the emanating data diffusion properties.

This simulation library has successfully been tested on

* Python 3.8 on Ubuntu 20.04
* Python 3.9 on MacOS 11.6

## 1. Setup

### 1.1 Clone the Repository

```
git clone https://git.tu-berlin.de/tobias.eichinger/dec-cf-sim.git
cd dec-cf-sim
```

### 1.2 Install `pip` and setup a virtual environment with `virtualenv`

On Linux/Ubuntu you can use the following commands.

```
# install the pip package manager for python3
sudo apt install python3-pip
# upgrade pip to the latest version
sudo python3 -m pip install --upgrade pip
# install the virtualenv package (to create a virtual environment)
sudo python3 -m pip install virtualenv
# create a virtual environment named "sim_venv"
sudo python3 -m virtualenv sim_venv
# activate the virtual environment "sim_venv"
source sim_venv/bin/activate
```

### 1.3 Install Python packages into the virtual environment

The installation is done line by line from the `./requirements.txt` file.
```
# install dependencies
cat requirements.txt | xargs -n 1 sudo -H python3 -m pip install
```

# 2. Demo

Run a demo simulation.

```
# enable execution of the ./demo.sh script
chmod +x ./demo.sh
# run the ./demo.sh script
./demo.sh
```

The demo simulates data dissemination between 8 test users with ratings defined in `./data/demo_data.csv`.
The demo produces an output directory `./data/snapshots/<simulation_output_dir>`, which contains a <userId>.csv for
every user in the input data. The output files hold the following columns.

[userId,itemId,rating,sim,agg_sim,sender,sim_to_sender,timestamp]

Note that the simulation produes cache files in the form of similarity dictionaries (`./data/similarity_dicts`) and
snapshot references (`./data/snapshot_ref`). Similarity dictionaries are serialized dictionaries that have (key,value)
pairs of the shape (user-user:epoch, sim(user,user)) to save recalculation time between runs on the same input data
and train-test random seed. The snapshot references are also dictionaries that hold (key,value) pairs of the shape
(userId,row-indices) as references to the row-indices of a user's entries in the input data to enable quick lookups.


# 3. Experiment Setup

You can setup your own dissemination simulation experiments. A simulation experiment is a full run of an algorithm such as './src/algorithms/DecCF.py'.
Have a look at the `./demo.sh` script for an example of how to run an algorithm file as a script properly.


### 3.1 Experiment Requirements

An experiment requires:

1. a user-item matrix file in coordinate format (see for instance `./data/demo_data.csv`).
1. the following minimal parameters:
	1. **N:** the number of contacts per epoch
	1. **T:** the number of epochs
	1. **graph_seed:** random seed for reproducibility of the execution graph
	1. **traintest_seed:** random seed for the reproducibility of the train-test split of the user-item matrix
	1. **payload_size:** max. number of profiles to send in a contact
	1. **mobility_model:** name of the mobility model to use, for instance "AssignNMobility"
	1. **percentile:** percentile of the similarity distribution of a user only above which a user sends data in a contact. Example: If the similarities of a user to all other users is uniformly distributed, then percentile=0.3 defines a similarity threshold of 0.3. The user only sends data in a contact, if their similarity is above this threshold.
1. a lot of RAM (currently).



### 3.2 Simulation Procedure

The simulation procedure is given in the `./src/algorithms/templates/DecAlgoTemplate.py` file.

Calling `<DecAlgoTemplate.fill_snapshots>` starts the simulation, where snapshots refers to the database snapshots of users. Snapshots are kept as pandas DataFrames and exported in *.csv format in `./data/snapshots` for later evaluation.

The `<DecAlgoTemplate.fill_snapshots>` method in detail:

1. Initialize simulation (snapshots, pairwise similarities ...)
	1. Iterate over all epochs t = 1,..,T and all users
		1. Collect the payload [i.e. information to be shared by the children with the parent] over all children of a parent node
		1. [OPTIONAL] Aggregate payload
		1. [OPTIONAL] Run a garbadge collection of the aggregated information in order to prevent for example duplicates
		1. [OPTIONAL] Update system parameters (theta, T (prune self.graph)); else use static parameters
		1. Add payloads to the respective users' snapshots (future_snapshots_dict[user])
	1. Save snapshots to <user>.csv files



## 4. Concepts

The simulation is based on an execution graph. The execution graph encodes consecutive encounters between all users given in the input user-item matrix (see 3.1).
The simulation graph is a Directed Acyclic Graph (DAG) and implements the class <DAG> in `./src/mobility_models/graph.py`.

*Example:* Let A, B, and C be users given in the input user-item matrix. The execution graph for a single contact between user A and C in epoch 1 is then
given by:

	B:0    A:0   C:0     ^
	|       | \_/ |      |
	|       | / \ |      | direction
	B:1    A:1   C:1     |



The idea is to separate functionalities as much as possible:

graphs: The management and graph (DAG) functionalities are provided here. The DAG encodes the encounters between the peers in a synchronous way,
that is for discrete (synchronous) time steps 0,1,2,.. interactions happen. This is a common abstraction in simulations. Every peer is represented by
several key value pairs over time. For example, the peer "A" will have the corresponding vertices with keys "A:0", "A:1",.. in the graph.
As the graph is a DAG (directed acyclic graph), there are no cycles (to be verified by the user of this library), i.e. going back in time.
The direction of the edges is from future to the past for reasons of computational simplicity. We would like to quickly get the calculation sub-DAG
in case that a particular snapshot is in need for calculation. Deducing the calculation sub-DAG is thus down-stream, even though the calculation of the
recommendations will happen upstream since the snapshots will have to be filled from past to future.

Every key-value pair denotes
* a user at some point t in time
It holds information on
* interactions (data exchanges) in the form of edges (set of keys in the past) --> value[0] as the first element of a duple
* snapshot information of the database that gets collected over time           --> value[1] as the second element of a duple


# Citation

This library has been used to obtain the results reported in a researcher paper with the following bibtex entry:

```
@inproceedings{Eichinger2023distributeddataminimization,
	author ={Eichinger, Tobias and K\"{u}pper, Axel},
	title ={Distributed Data Minimization for Decentralized Collaborative Filtering Systems},
	year ={2023},
	booktitle={Proceedings of the 24th International Conference On Distributed Computing And Networking (ICDCN)},
	publisher = {ACM},
	pages ={tbd},
	doi ={tbd},
}
```

If you use this library, please indicate this paper as a reference.
