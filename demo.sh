echo " ~~~~~~ START SIMULATION DEMO ~~~~~~~"
echo "This is a demo for the decentralized Collaborative Filtering simulation."
echo "We apply the algorithm './src/algorithms/DecCF.py' ... "
echo " ... with mobility model './src/mobility_models/NeighborhoodFormationNMobility.py' ... "
echo " ... to the demo data './data/demo_df.csv'. "

#######################
# I. SET PARAMETERS   #
#######################
# I.A EXPERIMENT PARAMETERS
experiment_name="demo_experiment"
N=3
T=5
graph_seed=123
traintest_seed=456
payload_size=5
mobility_model=AssignNMobility
percentile=0.1
# I.B EVALUATION PARAMETERS
experiment_dir="./data/snapshots/$experiment_name"
dataset_path="data/demo_df.csv"
CFformula="vanilla_CFformula_with_sim_to_sender"

# TEST
mean_centering="1"

#######################
# II. RUN EXPERIMENT  #
#######################
python3 ./src/algorithms/DecCF.py  ./data/demo_df.csv $experiment_name $N $T $graph_seed $traintest_seed $payload_size $mobility_model $percentile &
wait


echo " ~~~~~~END SIMULATION DEMO ~~~~~~ "
